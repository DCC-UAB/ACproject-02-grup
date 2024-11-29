import brain_utils as bu
import os
import cv2
import sklearn

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

### Ara ja tenim les imatges processades i guardades en el diccionari d'imatges
# SIFT:

sift = cv2.SIFT_create()

keypoints_descriptors = {}

for cancer_type, img_list in images.items():
    keypoints_descriptors[cancer_type] = []
    for img in img_list:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_descriptors[cancer_type].append(descriptors)

print(keypoints_descriptors)

### BoW - kmeans?


### histograma, X,y...


### SVC? no se si es viable amb tantes dimensions.1

