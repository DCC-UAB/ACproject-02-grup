import imageio.v2 as imageio
import os
import numpy as np
import random
import cv2

def is_top_view(image, margin=10, threshold=50):
    top_margin = image[:margin, :]
    bottom_margin = image[-margin:, :]
    left_margin = image[:, :margin]
    right_margin = image[:, -margin:]

    if (top_margin < threshold).all() and (bottom_margin < threshold).all() and \
       (left_margin < threshold).all() and (right_margin < threshold).all():
        return True
    return False

def filter_images(input_folder, output_folder, not_filtered_folder, num_images=-1, margin=10, threshold=50):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.exists(not_filtered_folder):
        os.makedirs(not_filtered_folder)

    filtered_count = 0
    images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)  # Shuffle images to get a random selection

    for filename in images:
        if num_images != -1 and filtered_count >= num_images:
            break

        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        not_filtered_path = os.path.join(not_filtered_folder, filename)

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
        else:
            imageio.imwrite(not_filtered_path, image)
            print(f"Image saved to: {not_filtered_path}")

    print(f"Number of images filtered from {input_folder}: {filtered_count}")
    return filtered_count

def filter_images_by_type(input_folders, output_base_folder, not_filtered_folder, margin=10, threshold=50):
    filter_images(input_folders["sans"], os.path.join(output_base_folder, "notumor"), not_filtered_folder, margin=margin, threshold=threshold)

    for folder_type in ["meningioma", "glioma", "pituitary"]:
        input_folder = input_folders[folder_type]
        output_folder = os.path.join(output_base_folder, f"{folder_type}")
        filter_images(input_folder, output_folder, not_filtered_folder, margin=margin, threshold=threshold)

def apply_proportions(input_folders, output_base_folder):
    notumor_filtered_count = len(os.listdir(os.path.join(output_base_folder, "notumor")))
    
    total_filtered_images = notumor_filtered_count / 0.50
    num_images_per_type = {
        "meningioma": int(total_filtered_images * 0.225),
        "glioma": int(total_filtered_images * 0.175),
        "pituitary": int(total_filtered_images * 0.10)
    }

    proportions_folder = os.path.join(output_base_folder, "proporcions")
    
    for folder_type, num_images in num_images_per_type.items():
        input_folder = os.path.join(output_base_folder, folder_type)
        output_folder = os.path.join(proportions_folder, folder_type)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)  # Shuffle images to get a random selection
        
        for i in range(min(num_images, len(images))):
            image_path = os.path.join(input_folder, images[i])
            output_path = os.path.join(output_folder, images[i])
            try:
                image = imageio.imread(image_path)
                imageio.imwrite(output_path, image)
                print(f"Image saved to: {output_path}")
            except Exception as e:
                print(f"Error processing the image {images[i]}: {e}")

def dos_tres(input_folder):
    images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image in images:
        image_path = os.path.join(input_folder, image)
        img = np.array(cv2.imread(image_path))
        if img is not None:
            tall = (2 * img.shape[0])//3
            img[tall:,:] = 0

            output_path = os.path.join(input_folder, f"FILTRE_{image}") #evitem sobreescriure millor :/
            cv2.imwrite(output_path, img)

input_folders = {
    "sans": "Brain Cancer/notumor",
    "meningioma": "Brain Cancer/meningioma",
    "glioma": "Brain Cancer/glioma",
    "pituitary": "Brain Cancer/pituitary"
}
output_base_folder = "Brain Cancer/filtre"
not_filtered_folder = os.path.join(output_base_folder, "not_filtered")

for folder in input_folders.values():
    if not os.path.exists(folder):
        print(f"Error: The input folder '{folder}' does not exist.")
        break
else:
    filter_images_by_type(input_folders, output_base_folder, not_filtered_folder, margin=12, threshold=50)

#apliquem les proporcions a partir de la carpeta de pacients sans (carpeta ja filtrada)
apply_proportions(input_folders, output_base_folder)


