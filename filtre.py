import imageio.v2 as imageio
import os
import numpy as np

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
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
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

# Configuració de carpetes
input_folder = "Carpeta amb les imatges a filtrar"
output_folder = "Carpeta on vols desar les imatges filtrades"

# Codi per mirar quin valor de marge va millor
try:
    num_images = int(input("Quantes imatges vols filtrar? (-1 per filtrar-les totes): "))
    if num_images < -1:
        raise ValueError("El número d'imatges no pot ser negatiu.")
except ValueError as e:
    print(f"Error: {e}")
else:
    if not os.path.exists(input_folder):
        print(f"Error: La carpeta d'entrada '{input_folder}' no existeix.")
    else:
        filter_images(input_folder, output_folder, num_images, margin=15, threshold=50)
