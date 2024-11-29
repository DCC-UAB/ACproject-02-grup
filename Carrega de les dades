"""Càrrega de les dades"""
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Configurar l'API
api = KaggleApi()
api.authenticate()

dataset_name = ""
download_path = ""
api.dataset_download_files(dataset_name, path=".", unzip=False)

# Descomprimir només la carpeta "Brain Cancer"
with zipfile.ZipFile(download_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if "/" in file:  
            zip_ref.extract(file, "")

# Eliminar el fitxer ZIP original per estalviar espai
os.remove(download_path)


