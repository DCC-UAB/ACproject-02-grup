import imageio.v2 as imageio
import os
import numpy as np
import random

def is_top_view(image, margin=10, threshold=50):
    top_margin = image[:margin, :]
    bottom_margin = image[-margin:, :]
    left_margin = image[:, :margin]
    right_margin = image[:, -margin:]

    if (top_margin < threshold).all() and (bottom_margin < threshold).all() and \
       (left_margin < threshold).all() and (right_margin < threshold).all():
        return True
    return False

def filter_images(input_folder, output_folder, num_images, margin=10, threshold=50):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filtered_count = 0
    images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Shuffle images to get a random selection

    for filename in images:
        if num_images != -1 and filtered_count >= num_images:
            break

        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            print(f"The image already exists in the output folder: {output_path}")
            continue

        try:
            image = imageio.imread(image_path, mode='L')
        except Exception as e:
            print(f"Error reading the image with imageio: {e}")
            continue

        if is_top_view(image, margin=margin, threshold=threshold):
            imageio.imwrite(output_path, image)
            print(f"Image saved to: {output_path}")
            filtered_count += 1

    print(f"Number of images filtered from {input_folder}: {filtered_count}")

def filter_images_by_type(input_folders, output_base_folder, num_images_per_type, margin=10, threshold=50):
    for folder_type, num_images in num_images_per_type.items():
        input_folder = input_folders[folder_type]
        output_folder = os.path.join(output_base_folder, f"{folder_type}_filtrat")
        filter_images(input_folder, output_folder, num_images, margin, threshold)

# Folder configuration
input_folders = {
    "sans": "Brain Cancer/notumor",
    "meningioma": "Brain Cancer/meningioma",
    "glioma": "Brain Cancer/glioma",
    "pituitari": "Brain Cancer/pituitary"
}
output_base_folder = "Brain Cancer"

# Percentages and number of images per type
num_images_per_type = {
    "sans": 20,
    "meningioma": 9,
    "glioma": 7,
    "pituitari": 4
}

for folder in input_folders.values():
    if not os.path.exists(folder):
        print(f"Error: The input folder '{folder}' does not exist.")
        break
else:
    filter_images_by_type(input_folders, output_base_folder, num_images_per_type, margin=15, threshold=50)
