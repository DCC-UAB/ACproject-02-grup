import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from tqdm import tqdm

# Configuració de les carpetes del dataset
base_path = "Brain Cancer/"
training_path = os.path.join(base_path, "Training")
testing_path = os.path.join(base_path, "Testing")
classes = ["glioma", "meningioma", "notumor", "pituitary"]
k = 100  # Número de paraules visuals (clústers)

# Funció per carregar imatges i etiquetes des de les carpetes
def load_images_and_labels_from_folder(folder_path, classes):
    images = []
    labels = []
    for idx, label in enumerate(classes):
        class_path = os.path.join(folder_path, label)
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Redimensionar imatges
                images.append(img)
                labels.append(idx)
    return np.array(images), np.array(labels)

# Extreure descriptors SIFT
def extract_sift_descriptors(images):
    sift = cv2.SIFT_create()
    all_descriptors = []
    for img in tqdm(images, desc="Extracting SIFT descriptors"):
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    return all_descriptors

# Crear un diccionari visual amb K-Means
def create_visual_dictionary(all_descriptors, k):
    descriptors = np.vstack(all_descriptors)
    print("Clustering descriptors into", k, "visual words...")
    kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
    kmeans.fit(descriptors)
    return kmeans

# Crear histogrames BoVW
def create_bovw_histograms(images, kmeans):
    sift = cv2.SIFT_create()
    histograms = []
    for img in tqdm(images, desc="Creating BoVW histograms"):
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(kmeans.n_clusters))
    return np.array(histograms)

# Pipeline principal
def main():
    # Carregar dades d'entrenament i test
    print("Loading training images...")
    train_images, train_labels = load_images_and_labels_from_folder(training_path, classes)
    print("Loading testing images...")
    test_images, test_labels = load_images_and_labels_from_folder(testing_path, classes)
    
    # Extreure descriptors SIFT per entrenament
    print("Extracting SIFT descriptors from training data...")
    train_descriptors = extract_sift_descriptors(train_images)
    
    # Crear diccionari visual
    kmeans = create_visual_dictionary(train_descriptors, k)
    
    # Crear histogrames BoVW
    print("Creating BoVW histograms for training data...")
    train_histograms = create_bovw_histograms(train_images, kmeans)
    print("Creating BoVW histograms for testing data...")
    test_histograms = create_bovw_histograms(test_images, kmeans)
    
    # Entrenar un classificador (SVM)
    print("Training SVM classifier...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(train_histograms, train_labels)
    
    # Avaluar el model
    print("Evaluating the model...")
    test_predictions = svm.predict(test_histograms)
    print("Accuracy:", accuracy_score(test_labels, test_predictions))
    print(classification_report(test_labels, test_predictions, target_names=classes))

if __name__ == "__main__":
    main()

