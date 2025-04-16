from sklearn import datasets
from sklearn.utils import Bunch
import pandas as pd
import numpy as np


def load_sklearn_datasets():
    """Carga todos los datasets disponibles de scikit-learn"""
    datasetsList = {
        'iris': datasets.load_iris(),
        'diabetes': datasets.load_diabetes(),
        'digits': datasets.load_digits(),
        'wine': datasets.load_wine(),
        'breast_cancer': datasets.load_breast_cancer(),
        'california_housing': datasets.fetch_california_housing(),
        'olivetti_faces': datasets.fetch_olivetti_faces(),
    }

    # Asegurar atributos mínimos para todos los datasets
    for name, data in datasetsList.items():
        if not hasattr(data, 'feature_names'):
            data.feature_names = [f"Feature {i}" for i in range(data.data.shape[1])]
        if not hasattr(data, 'target_names'):
            data.target_names = np.unique(data.target).astype(str)

    return datasetsList


def get_dataset_names():
    """Devuelve los nombres de los datasets disponibles"""
    return list(load_sklearn_datasets().keys())


def get_dataset(name):
    """Obtiene un dataset por nombre"""
    return load_sklearn_datasets().get(name)


def load_custom_dataset(file_path, target_column=-1):
    try:
        data = pd.read_csv(file_path)

        if target_column < 0:
            target_column = data.columns[target_column]

        target = data[target_column].values

        unique_classes = np.unique(target)
        target_names = [f"Class_{cls}" for cls in unique_classes]  # O usar str(cls)
        features = data.drop(columns=[target_column]).values

        # Generar nombres de características si no existen
        feature_names = data.drop(columns=[target_column]).columns.tolist() or \
                        [f"Feature {i}" for i in range(features.shape[1])]

        return Bunch(
            data=features,
            target=target,
            feature_names=feature_names,
            target_names=target_names,
            DESCR=f"Custom dataset from {file_path}",
            filename=file_path
        )
    except Exception as e:
        raise ValueError(f"Error cargando dataset: {str(e)}")