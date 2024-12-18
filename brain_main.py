import brain_utils as bu
import os
import cv2
import sklearn
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Processament de les dades, resize a 256x256 i ordenament a les seves respectives llistes
dir = './Brain Cancer/filtre'
print(os.listdir(dir))

images = {}

for cancer_type in os.listdir(dir):  # pels 3 tipus de cancer
    subfolder_path = os.path.join(dir, cancer_type)
    imglist = os.listdir(subfolder_path)  # llista de les imatges dins de cada tipus

    images[cancer_type] = []  # creem llista d'imatges al dict

    for img in imglist:
        img_gray = bu.resize_image(os.path.join(subfolder_path, img))  # fem el processament i afegim a respectiva llista
        images[cancer_type].append(img_gray)

# Extreure descriptors SIFT
dense_sift_features = {}
step_size = 8  # Ajusta la densitat segons calgui

for cancer_type, img_list in images.items():
    dense_sift_features[cancer_type] = []
    for img in img_list:
        keypoints, descriptors = bu.denseSIFT_mask(img, step_size=step_size)  # te una mascara binaria
        dense_sift_features[cancer_type].append(descriptors)  # Guarda descriptors

# Configuració de carpetes i classes
base_path = './Brain Cancer/filtre'
data_path = base_path
classes = ["glioma", "meningioma", "notumor", "pituitary"]
k = 50  # Número de clústers

# Carregar dades
print("Carregant imatges i etiquetes...")
images, labels = bu.load_images_and_labels_from_folder(data_path, classes)

# Dividir les dades en entrenament i test
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# Extreure descriptors SIFT per entrenament
train_descriptors = []
for img in train_images:
    _, descriptors = bu.denseSIFT_mask(img)
    if descriptors is not None:
        train_descriptors.append(descriptors)

# Crear diccionari visual amb K-Means i provar diferents valors de k
k_values = [40, 50, 75, 100]
accuracies = []

for k in k_values:
    print(f"Entrenant amb k={k}...")
    kmeans = bu.create_visual_dictionary(train_descriptors, k)
    train_histograms = bu.create_bovw_histograms(train_descriptors, kmeans)
    test_descriptors = [bu.denseSIFT_mask(img)[1] for img in test_images]
    test_histograms = bu.create_bovw_histograms(test_descriptors, kmeans)
    
    # Entrenar el model SVM
    svm = SVC(kernel='rbf', random_state=42, probability=True)  # Habilitar probabilitats
    svm.fit(train_histograms, train_labels)
    
    # Obtenir prediccions i calcular la precisió
    test_predictions = svm.predict(test_histograms)
    accuracy = accuracy_score(test_labels, test_predictions)
    accuracies.append(accuracy)

# Mostrar gràfic dels resultats de precisió per a diferents k
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Nombre de clústers (k)')
plt.ylabel('Precisió')
plt.title('Rendiment de KMeans amb diferents valors de k')
plt.show()

# Escollir el millor valor de k (el que doni millor precisió)
best_k = k_values[accuracies.index(max(accuracies))]
print(f"El millor valor de k és: {best_k}")

# Crear el diccionari visual amb el millor k
kmeans = bu.create_visual_dictionary(train_descriptors, best_k)
train_histograms = bu.create_bovw_histograms(train_descriptors, kmeans)
test_histograms = bu.create_bovw_histograms(test_descriptors, kmeans)

# Entrenar el model amb el millor k
svm = SVC(kernel='rbf', random_state=42, probability=True)
svm.fit(train_histograms, train_labels)

# Obtenir les probabilitats de les prediccions per a calcular la ROC
y_test_bin = label_binarize(test_labels, classes=classes)
y_pred_prob = svm.predict_proba(test_histograms)

# Calcular i plottejar la ROC per a cada classe
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC per a {classes[i]}')
    plt.legend(loc="lower right")
    plt.show()

# Avaluar el model amb el millor k
test_predictions = svm.predict(test_histograms)
print("Accuracy:", accuracy_score(test_labels, test_predictions))
print(classification_report(test_labels, test_predictions, target_names=classes))
