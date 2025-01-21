from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def estimate_depth_and_plot(img_path, model_checkpoint):
    """
    Function to estimate depth from an image, output the depth map as a numpy array,
    and display the original vs depth map side-by-side.
    
    Args:
    - img_path (str): Path to the input image.
    - model_checkpoint (str): Hugging Face model checkpoint for depth estimation.
    
    Returns:
    - np.ndarray: Depth map as a numpy array.
    """
    # Initialize depth estimation pipeline
    pipe = pipeline("depth-estimation", model=model_checkpoint, device=0)

    # Load image
    image = Image.open(img_path)

    # Perform depth estimation
    predictions = pipe(image)

    # Get depth output as a PIL image
    depth_image = predictions["depth"]

    # Convert depth PIL image to a NumPy array
    depth_array = np.array(depth_image)

    # Plot the original and depth images
    # plt.figure(figsize=(12, 6))

    # # Original image
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title("Original Image")
    # plt.axis("off")

    # # Depth map
    # plt.subplot(1, 2, 2)
    # plt.imshow(depth_array, cmap="viridis")
    # plt.title("Depth Map")
    # plt.axis("off")

    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    return depth_array

# Example usage
img_path = r"F:\Deep Learning\computer_vision\Autonomous_driving\data_object_image_2\testing\image_2\000000.png"
model_checkpoint = "depth-anything/Depth-Anything-V2-base-hf"

depth_array = estimate_depth_and_plot(img_path, model_checkpoint)
print("Depth array shape:", depth_array.shape)

