import brain_utils as bu
import os
import cv2
import sklearn
import time

t0 = time.time()

### Primer processament de les dades, resize a 256x256 i ordenament a les seves respectives llistes
dir = './Brain Cancer'
print(os.listdir(dir))

images = {}

for cancer_type in os.listdir(dir):                             # pels 3 tipus de cancer
    subfolder_path = os.path.join(dir, cancer_type)
    imglist = os.listdir(subfolder_path)                        # llista de les imatges dins de cada tipus

    images[cancer_type] = []                                                  # creem llista d'imatges al dict

    # print(imglist)
    for img in imglist:
        # print(img)
        img_gray = bu.resize_image(os.path.join(subfolder_path, img))      # fem el processament i afegim a respectiva llista
        images[cancer_type].append(img_gray)
        # print(images)

t1 = time.time()
print('Imatges processades en:', t1-t0)
### Ara ja tenim les imatges processades i guardades en el diccionari d'imatges
# SIFT:

dense_sift_features = {}
step_size = 8  # Ajusta la densitat segons calgui

for cancer_type, img_list in images.items():
    dense_sift_features[cancer_type] = []
    for img in img_list:
        # print('ei')
        keypoints, descriptors = bu.denseSIFT_mask(img, step_size=step_size) # te una mascara binaria, ignora tots els pixels que estiguin a 0
        dense_sift_features[cancer_type].append(descriptors)  # Guarda descriptors

print(dense_sift_features)

t2 = time.time()
print('SIFT descriptors creats en:', t2-t1)

### BoW - kmeans?


### histograma, X,y...


### SVC? no se si es viable amb tantes dimensions. --> rbf

