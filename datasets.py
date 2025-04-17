from sklearn import datasets
from sklearn.utils import Bunch
import pandas as pd
import numpy as np


def load_sklearn_datasets():
    """
    Carga y prepara varios datasets de scikit-learn con estructura estandarizada.
    Returns:
        dict: Diccionario con nombres de datasets como claves y objetos Bunch como valores.
              Cada dataset garantiza tener feature_names y target_names.
    """
    datasetsList = {
        'iris': datasets.load_iris(),
        'diabetes': datasets.load_diabetes(),
        'digits': datasets.load_digits(),
        'wine': datasets.load_wine(),
        'breast_cancer': datasets.load_breast_cancer(),
        'california_housing': datasets.fetch_california_housing(),
        'olivetti_faces': datasets.fetch_olivetti_faces(),
    }

    for name, data in datasetsList.items():
        if not hasattr(data, 'feature_names'):
            data.feature_names = [f"Feature {i}" for i in range(data.data.shape[1])]
        if not hasattr(data, 'target_names'):
            data.target_names = np.unique(data.target).astype(str)

    return datasetsList


def get_dataset_names():
    """
    Obtiene la lista de nombres de datasets disponibles.
    Returns:
        list: Lista de strings con los nombres de los datasets cargados.
    """
    return list(load_sklearn_datasets().keys())


def get_dataset(name):
    """
    Obtiene un dataset específico por su nombre.
    Args:
        name (str): Nombre del dataset a recuperar.
    Returns:
        Bunch: Objeto dataset correspondiente, o None si no existe.
    """
    return load_sklearn_datasets().get(name)


def load_custom_dataset(file_path, target_column=-1):
    """
    Carga un dataset personalizado desde un archivo CSV.
    Args:
        file_path (str): Ruta al archivo CSV.
        target_column (int): Índice de la columna objetivo (por defecto la última).
    Returns:
        Bunch: Objeto dataset con estructura similar a los de scikit-learn.
    """
    try:
        data = pd.read_csv(file_path)

        if target_column < 0:
            target_column = data.columns[target_column]

        target = data[target_column].values

        unique_classes = np.unique(target)
        target_names = [f"Class_{cls}" for cls in unique_classes]
        features = data.drop(columns=[target_column]).values

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