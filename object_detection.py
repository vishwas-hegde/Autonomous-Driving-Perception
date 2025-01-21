import os
import cv2
from ultralytics import YOLO

# Load the COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

def process_images_to_video(input_folder, output_video_path, fps=30):
    """
    Process images in the input folder using YOLO and save as a video.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.
    """
    # Get a sorted list of image paths
    image_paths = sorted([
        os.path.join(input_folder, img) for img in os.listdir(input_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not image_paths:
        print("No images found in the folder.")
        return

    # Read the first image to determine video frame size
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        # Predict with YOLOv8
        results = model(img_path)

        # Render the annotated image
        annotated_frame = results[0].plot()

        # Write the frame to the video
        video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved at {output_video_path}")

# Example usage
input_folder = r"F:\Deep Learning\computer_vision\Autonomous_driving\kitti_ds\0047"  # Replace with your image folder path
output_video_path = r"F:\Deep Learning\computer_vision\Autonomous_driving\output\obj_detection.mp4"       # Replace with desired output video path
fps = 30  # Adjust as needed

process_images_to_video(input_folder, output_video_path, fps)
