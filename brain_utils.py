import cv2
import os
import numpy as np
import sklearn

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



