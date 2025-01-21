import os
import shutil
from pathlib import Path

def organize_images_by_dataset(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the input folder
    input_folder_path = Path(input_folder)
    image_files = [f for f in input_folder_path.glob('*') if f.is_file()]

    # Dictionary to store images by their dataset number
    images_by_dataset = {}

    # Iterate through all image files
    for image_file in image_files:
        # Extract dataset number (assuming it's the 4 digits in the middle, e.g., 0047)
        filename = image_file.name
        dataset_number = filename[17:21]  # Extract the dataset number from the filename

        # Add image to corresponding dataset number
        if dataset_number not in images_by_dataset:
            images_by_dataset[dataset_number] = []

        images_by_dataset[dataset_number].append(image_file)

    # Create subfolders for each dataset number and copy corresponding images
    for dataset_number, images in images_by_dataset.items():
        # Create a folder for the dataset number
        dataset_folder = os.path.join(output_folder, dataset_number)
        os.makedirs(dataset_folder, exist_ok=True)

        # Copy each image to the corresponding dataset folder
        for image in images:
            shutil.copy(image, dataset_folder)
            print(f"Copied {image.name} to {dataset_folder}")

if __name__ == "__main__":
    # Define input and output folder paths
    input_folder = r"C:\Users\vishw\Downloads\data_depth_selection\depth_selection\val_selection_cropped\image"  # Replace with your input folder path
    output_folder = r"F:\Deep Learning\computer_vision\Autonomous_driving\kitti_ds"  # Replace with your desired output folder path

    # Call the function to organize the images
    organize_images_by_dataset(input_folder, output_folder)
