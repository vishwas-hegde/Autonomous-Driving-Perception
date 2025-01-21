
import os
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

def estimate_depth(image, model_checkpoint):
    """
    Estimate the depth map for the input image using the depth model.

    Args:
    - image (PIL.Image): Input image.
    - model_checkpoint (str): Hugging Face model checkpoint for depth estimation.

    Returns:
    - np.ndarray: Depth map as a numpy array.
    """
    # Perform depth estimation
    predictions = pipe(image)

    # Get depth output as a PIL image
    depth_image = predictions["depth"]

    # Convert depth PIL image to a NumPy array
    depth_array = np.array(depth_image)

    return depth_array

def create_depth_video(input_folder, model_checkpoint, output_video_path, frame_size=(640, 480), fps=10):
    """
    Create a video from depth maps of images in a folder.

    Args:
    - input_folder (str): Folder containing images.
    - model_checkpoint (str): Hugging Face model checkpoint for depth estimation.
    - output_video_path (str): Path to save the output video.
    - frame_size (tuple): Size of the video frames (width, height).
    - fps (int): Frames per second for the video.
    """
    # Get all image files from the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    # Initialize video writer (fourcc, width, height, fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Process each image in the sorted list
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)

        # Load image
        image = Image.open(img_path)

        # Estimate depth map
        depth_array = estimate_depth(image, model_checkpoint)

        # Resize depth map to match the frame size
        depth_resized = cv2.resize(depth_array, frame_size)

        # Normalize depth to 0-255 range for visualization
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # Write the depth map as a frame in the video
        out.write(depth_colored)

        print(f"Processed and added frame for {image_file}")

    # Release the video writer and finalize the video
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
input_folder = r"F:\Deep Learning\computer_vision\Autonomous_driving\kitti_ds\0047"
# model_checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
model_checkpoint = "Intel/zoedepth-nyu-kitti"
output_video_path = r"F:\Deep Learning\computer_vision\Autonomous_driving\output\depth_output_zoe_video.avi"
# Initialize depth estimation pipeline
pipe = pipeline("depth-estimation", model=model_checkpoint, device=0)
# Create video from depth maps
create_depth_video(input_folder, model_checkpoint, output_video_path)
