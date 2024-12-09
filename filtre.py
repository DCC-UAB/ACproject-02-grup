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
    random.shuffle(images)  # Barreja les imatges per obtenir una selecció aleatòria

    for filename in images:
        if num_images != -1 and filtered_count >= num_images:
            break

        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            print(f"La imatge ja existeix a la carpeta de sortida: {output_path}")
            continue

        try:
            image = imageio.imread(image_path, mode='L')
        except Exception as e:
            print(f"Error llegint la imatge amb imageio: {e}")
            continue

        if is_top_view(image, margin=margin, threshold=threshold):
            imageio.imwrite(output_path, image)
            print(f"Imatge guardada a: {output_path}")
            filtered_count += 1

    return filtered_count

def filter_images_by_type(input_folders, output_folder, margin=10, threshold=50):
    sans_folder = input_folders["sans"]
    num_sans_images = filter_images(sans_folder, output_folder, -1, margin, threshold)
    
    total_images = num_sans_images * 2  # El 50% de les dades finals són pacients sans
    num_images_per_type = {
        "meningioma": int(total_images * 0.225),
        "glioma": int(total_images * 0.175),
        "pituitari": int(total_images * 0.10)
    }

    # Filtra les altres carpetes basant-se en els nombres calculats
    for folder in ["meningioma", "glioma", "pituitari"]:
        input_folder = input_folders[folder]
        filter_images(input_folder, output_folder, num_images_per_type[folder], margin, threshold)

# Configuració de carpetes
input_folders = {
    "sans": " ",
    "meningioma": " ",
    "glioma": " ",
    "pituitari": " "
}
output_folder = " "

for folder in input_folders.values():
    if not os.path.exists(folder):
        print(f"Error: La carpeta d'entrada '{folder}' no existeix.")
        break
else:
    filter_images_by_type(input_folders, output_folder, margin=15, threshold=50)
