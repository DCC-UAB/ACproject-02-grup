import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

"""DOCUMENT DE FUNCIONS, RES DE PIPELINES. AQUÍ POSEM LES FUNCIONS PER CRIDAR-LES EN EL 'MAIN' NOSTRE --> BRAIN-MAIN.PY"""

def resize_image(image_path, target_size=(256, 256)):

    """
    Funció de PSIV de tota la vida, agafem l'imatge i redimensionem +BW per si de cas
    
    :param image_path: cami de l'imatge
    :param target_size: mida final de la foto
    
    """
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    return img_gray


def denseSIFT_mask(image, step_size=8):
    """
    Aplica Dense SIFT sobre una imatge usant una mascara +1 per ignorar zones no rellevants

    :param image: Imatge (numpy array) en escala de grisos.
    :param step_size: Separació entre punts clau (int).
    :return: Punts clau (keypoints) i descriptors SIFT.
    """
    # Crea la màscara automàticament (ignora el negre)
    _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)  # Detecta píxels no negres (1 o més)

    # Defineix els punts clau en graella
    keypoints = [
        cv2.KeyPoint(x, y, step_size)
        for y in range(0, image.shape[0], step_size)
        for x in range(0, image.shape[1], step_size)
        if mask[y, x] > 0  # Inclou només punts dins de la màscara
    ]

    # Crea l'objecte SIFT
    sift = cv2.SIFT_create()

    # Calcula descriptors pels punts clau dins de la màscara
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors


# Carregar imatges i etiquetes
def load_images_and_labels_from_folder(folder_path, classes):
    images = []
    labels = []
    for idx, label in enumerate(classes):
        class_path = os.path.join(folder_path, label)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = resize_image(img_path)  # Usar la funció existent
            if img is not None:
                images.append(img)
                labels.append(idx)
    return np.array(images), np.array(labels)

# Crear un diccionari visual amb K-Means
def create_visual_dictionary(all_descriptors, k):
    descriptors = np.vstack(all_descriptors)
    print("Clustering descriptors into", k, "visual words...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(descriptors)
    return kmeans

# Crear histogrames BoVW
def create_bovw_histograms(images, kmeans):
    sift = cv2.SIFT_create()
    histograms = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(kmeans.n_clusters))
    return np.array(histograms)



