import cv2
import os
import numpy as np
import sklearn

"""DOCUMENT DE FUNCIONS, RES DE PIPELINES. AQUÍ POSEM LES FUNCIONS PER CRIDAR-LES EN EL 'MAIN' NOSTRE --> BRAIN-MAIN.PY"""

def resize_image(image_path, target_size=(256, 256)):

    """Funció de PSIV de tota la vida, agafem l'imatge i redimensionem +BW per si de cas"""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    return img_gray


