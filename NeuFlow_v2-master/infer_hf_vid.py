import torch
from glob import glob
import os
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from data_utils import flow_viz

image_width = 768
image_height = 432

def get_cuda_image(image):
    image = cv2.resize(image, (image_width, image_height))
    image = torch.from_numpy(image).permute(2, 0, 1).half()
    return image[None].cuda()

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def extract_frames_from_video(video_path, max_frames=150):
    """Extract up to `max_frames` frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()
    # print(len(frames), "frames extracted from video")
    # exit(0)
    return frames

# Video path and output directory
video_path = r'F:\Deep Learning\computer_vision\Autonomous_driving\back3.mp4'
vis_path = 'test_results/'

# Extract frames from video
frames = extract_frames_from_video(video_path, max_frames=150)

device = torch.device('cuda')

model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(device)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
        delattr(m, "norm1")  # remove batchnorm
        delattr(m, "norm2")  # remove batchnorm
        m.forward = m.forward_fuse  # update forward

model.eval()
model.half()

model.init_bhwd(1, image_height, image_width, 'cuda')

if not os.path.exists(vis_path):
    os.makedirs(vis_path)
output_video_path = 'output_optical_flow2.mp4'
# Initialize VideoWriter
fps = 30  # Frames per second for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (image_width, image_height))

# Process frames in pairs
for i in range(len(frames) - 1):
    print(f"Processing frame pair {i} and {i + 1}")

    image_0 = get_cuda_image(frames[i])
    image_1 = get_cuda_image(frames[i + 1])

    file_name = f"frame_{i:04d}.png"

    with torch.no_grad():
        flow = model(image_0, image_1)[-1][0]
        flow = flow.permute(1, 2, 0).cpu().numpy()
        flow = flow_viz.flow_to_image(flow)

        image_0_resized = cv2.resize(frames[i], (image_width, image_height))
        cv2.imwrite(os.path.join(vis_path, file_name), flow)
    
    out_video.write(flow)

# Release the VideoWriter
out_video.release()

print(f"Optical flow video saved to {output_video_path}")
