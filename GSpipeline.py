import brain_utils as bu
import os
import time
import numpy as np
import pandas as pd
# import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

# Configuració inicial
base_dirs = ['./Brain Cancer/filtre','./Brain Cancer/filtre/proporcions']
k_values = [64, 128, 256]  # Visual words
svm_param_grid = {'kernel': ['rbf', 'poly'], 'C': [0.1, 1, 10], 'gamma': ['auto']}
classes = ["glioma", "meningioma", "notumor", "pituitary"]
step_sizes = [8, 16]  # Afegim step_size

# Mètrica personalitzada
scorer = make_scorer(accuracy_score)

# Funció principal per processar un directori
def process_directory(base_path, results):
    print(f"\nProcessant directori: {base_path}")
    data_path = {c: os.path.join(base_path, c) for c in classes}

    # Carregar dades
    images, labels = bu.load_images_and_labels_from_dict(data_path, classes)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    for step_size in step_sizes:
        print(f"\nCalculant descriptors SIFT amb step_size={step_size}...")
        train_descriptors = [bu.denseSIFT_mask(img, step_size=step_size)[1] for img in train_images]
        test_descriptors = [bu.denseSIFT_mask(img, step_size=step_size)[1] for img in test_images]

        # Iteració sobre k per a diccionaris visuals
        for k in k_values:
            print(f"\nCreant diccionari visual per k={k}, step_size={step_size}")
            kmeans = bu.create_visual_dictionary(train_descriptors, k)

            # Transformar dades en histogrames de BOVW
            train_histograms = bu.create_bovw_histograms(train_descriptors, kmeans)
            test_histograms = bu.create_bovw_histograms(test_descriptors, kmeans)

            # Construir pipeline: Normalització + SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                ('svm', SVC(probability=True, random_state=42))
            ])

            # GridSearch amb pipeline
            print(f"Executant GridSearch per k={k}, step_size={step_size}...")
            start_time = time.time()
            grid_search = GridSearchCV(
                pipeline, param_grid={'svm__' + k: v for k, v in svm_param_grid.items()},
                scoring=scorer, cv=3, verbose=1, n_jobs=6
            )
            grid_search.fit(train_histograms, train_labels)

            # Resultats del millor model
            best_model = grid_search.best_estimator_
            test_predictions = best_model.predict(test_histograms)

            accuracy = accuracy_score(test_labels, test_predictions)
            recall = recall_score(test_labels, test_predictions, average='macro')
            f1 = f1_score(test_labels, test_predictions, average='macro')

            # Guardar el model entrenat
            model_file = os.path.join(base_path, f"svm_model_k{k}_step{step_size}.joblib")
            dump(best_model, model_file)

            # Registrar resultats
            elapsed_time = time.time() - start_time
            results.append({
                'Directory': base_path, 'k': k, 'Step Size': step_size, **grid_search.best_params_,
                'Accuracy': accuracy, 'Recall': recall, 'F1-Score': f1,
                'Time (s)': elapsed_time, 'Model Path': model_file
            })
            print(f"Resultat -> k={k}, step_size={step_size}, Acc: {accuracy:.4f}, Time: {elapsed_time:.2f}s")

# Executar el processament
if __name__ == "__main__":
    all_results = []
    for base_dir in base_dirs:
        process_directory(base_dir, all_results)
        print(all_results)

    # Guardar resultats finals
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('svm_pipeline_results.csv', index=False)
    print("\nResultats guardats a 'svm_pipeline_results.csv'")
