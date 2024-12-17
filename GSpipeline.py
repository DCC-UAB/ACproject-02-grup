import brain_utils as bu
import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix, make_scorer
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# Configuració inicial
base_dirs = ['./Brain Cancer/filtre', './Brain Cancer/proporcions']
k_values = [64, 128, 256, 512]  # Visual words
kernels = ['rbf', 'poly']
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto']
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Mètrica personalitzada
scorer = make_scorer(accuracy_score)

# Processament principal
results = []
for base_path in base_dirs:
    print(f"\nProcessant directori: {base_path}")

    # Carregar dades
    data_path = {c: os.path.join(base_path, c) for c in classes}
    images, labels = bu.load_images_and_labels_from_dict(data_path, classes)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Precomputació de descriptors SIFT
    print("Calculant descriptors SIFT...")
    train_descriptors = [bu.denseSIFT_mask(img, step_size=8)[1] for img in train_images]
    test_descriptors = [bu.denseSIFT_mask(img, step_size=8)[1] for img in test_images]

    # Iterar per valors de k per als diccionaris visuals
    for k in k_values:
        print(f"\nCreant diccionari visual per k={k}")
        kmeans = bu.create_visual_dictionary(train_descriptors, k)
        train_histograms = bu.create_bovw_histograms(train_descriptors, kmeans)
        test_histograms = bu.create_bovw_histograms(test_descriptors, kmeans)

        # Definir el model SVM i hiperparàmetres per a GridSearch
        svm = SVC(probability=True, random_state=42)
        param_grid = {
            'kernel': kernels,
            'C': C_values,
            'gamma': gamma_values
        }

        # Inicialitzar GridSearchCV
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            scoring=scorer,
            cv=3,  # Validació creuada amb 3 particions
            verbose=1,
            n_jobs=-1
        )

        # Entrenar GridSearch
        print(f"Executant GridSearch per k={k}...")
        start_time = time.time()
        grid_search.fit(train_histograms, train_labels)

        # Prediccions amb el millor model
        best_model = grid_search.best_estimator_
        test_predictions = best_model.predict(test_histograms)

        # Mètriques
        accuracy = accuracy_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions, average='macro')
        f1 = f1_score(test_labels, test_predictions, average='macro')

        # Confusion Matrix
        cm = confusion_matrix(test_labels, test_predictions, labels=classes)
        cm_file = f"confusion_matrix_k{k}.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix (k={k}, Best Params={grid_search.best_params_})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(cm_file)
        plt.close()

        # Guardar resultats
        elapsed_time = time.time() - start_time
        results.append({
            'Directory': base_path, 'k': k, **grid_search.best_params_,
            'Accuracy': accuracy, 'Recall': recall, 'F1-Score': f1, 'Time (s)': elapsed_time,
            'Confusion Matrix': cm_file
        })
        print(f"Resultat -> k={k}, Best Params={grid_search.best_params_}, Acc: {accuracy:.4f}, Time: {elapsed_time:.2f}s")

# Guardar resultats en CSV
results_df = pd.DataFrame(results)
results_df.to_csv('svm_pipeline_results.csv', index=False)
print("\nResultats guardats a 'svm_pipeline_results.csv'")
