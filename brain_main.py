import brain_utils as bu
import os
import cv2
import sklearn
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

'''t0 = time.time()

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



### SVC? no se si es viable amb tantes dimensions. --> rbf'''

# Configuració de carpetes i classes
base_path = './Brain Cancer'
training_path = f"{base_path}/Training"
testing_path = f"{base_path}/Testing"
classes = ["glioma", "meningioma", "notumor", "pituitary"]
k = 50  # Número de clústers

# Processament principal
t0 = time.time()

# Carregar dades
print("Carregant imatges d'entrenament...")
train_images, train_labels = bu.load_images_and_labels_from_folder(training_path, classes)
print("Carregant imatges de test...")
test_images, test_labels = bu.load_images_and_labels_from_folder(testing_path, classes)

t1 = time.time()
print(f"Dades carregades i processades en: {t1 - t0:.2f} segons")

# Extreure descriptors SIFT per entrenament
print("Extreient descriptors Dense SIFT...")
train_descriptors = []
for img in train_images:
    _, descriptors = bu.denseSIFT_mask(img)
    if descriptors is not None:
        train_descriptors.append(descriptors)

# Crear diccionari visual
print("Creant diccionari visual amb K-Means...")
kmeans = bu.create_visual_dictionary(train_descriptors, k)

# Crear histogrames BoVW
print("Creant histogrames BoVW per a dades d'entrenament...")
train_histograms = bu.create_bovw_histograms(train_images, kmeans)
print("Creant histogrames BoVW per a dades de test...")
test_histograms = bu.create_bovw_histograms(test_images, kmeans)

# Aquí podrías agregar la lógica para entrenar el SVM o cualquier otro modelo
# per a predir i avaluar resultats.

t2 = time.time()
print(f"Pipeline completat en: {t2 - t0:.2f} segons")

# Entrenar un classificador (SVM)
print("Training SVM classifier...")
#svm = SVC(kernel='rbf', random_state=42)
svm = SVC(kernel='linear', random_state=42)

svm.fit(train_histograms, train_labels)

# Avaluar el model
print("Evaluating the model...")
test_predictions = svm.predict(test_histograms)
print("Accuracy:", accuracy_score(test_labels, test_predictions))
print(classification_report(test_labels, test_predictions, target_names=classes))
