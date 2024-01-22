from PIL import Image
import os

folders = ["BLA/0001-1000", "EBO/0001-1000", "LYT/0001-1000", "NGS/0001-1000", "PLM/0001-1000", "PMO/0001-1000"]
output_path = "output_square.jpg"  # Adjust the output path as needed

# Create a list to store images from each class
class_images = []

# Iterate through each class folder
for folder in folders:
    class_folder_path = os.path.join("/home/mikylab/datasets/cyto_full/bone_marrow_cell_dataset/classes6/testing", folder)
    # Take the first 5 images from each class
    image_files = os.listdir(class_folder_path)[:5]
    # Load and append each image to the list
    class_images.append([Image.open(os.path.join(class_folder_path, img)) for img in image_files])
    # Calculate the size of the square canvas
    canvas_size = (len(folders) * 100, 500)  # Adjust dimensions as needed
    # Create a new image with a white background
    canvas = Image.new("RGB", canvas_size, "white")
    # Paste each class column onto the canvas
    for i, images in enumerate(class_images):
        for j, img in enumerate(images):
            # Calculate position for pasting each image
            x_offset = i * 100
            y_offset = j * 100
            canvas.paste(img, (x_offset, y_offset))
            # Save the final square image
            canvas.save(output_path)

