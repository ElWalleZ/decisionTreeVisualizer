import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

import joblib

def entrenar_arbol_decision(dataset, max_depth=3):
    try:
        X = dataset.data
        y = dataset.target

        # Verificar tipo de problema
        if y.ndim > 1:
            raise ValueError("Target multidimensional no soportado")

        es_clasificacion = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 10

        # Entrenar modelo adecuado
        if es_clasificacion:
            modelo = DecisionTreeClassifier(max_depth=max_depth)
            metric_name = "Precisión"
        else:
            modelo = DecisionTreeRegressor(max_depth=max_depth)
            metric_name = "R²"

        modelo.fit(X, y)

        # Generar texto del árbol
        if es_clasificacion:
            class_names = getattr(dataset, 'target_names', [str(c) for c in modelo.classes_])
            texto_arbol = export_text(
                modelo,
                feature_names=list(getattr(dataset, 'feature_names', [])),
                class_names=class_names
            )
            metrica = accuracy_score(y, modelo.predict(X))
        else:
            texto_arbol = export_text(
                modelo,
                feature_names=list(getattr(dataset, 'feature_names', []))
            )
            metrica = r2_score(y, modelo.predict(X))

        return {
            'modelo': modelo,
            'texto_arbol': texto_arbol,
            'metrica': metrica,
            'tipo': 'clasificacion' if es_clasificacion else 'regresion',
            'metric_name': metric_name
        }
    except Exception as e:
        raise ValueError(f"Error entrenando modelo: {str(e)}")


def guardar_modelo(modelo, directorio):
    joblib.dump(modelo, os.path.join(directorio, 'modelo_arbol.pkl'))


def cargar_modelo(directorio):
    return joblib.load(os.path.join(directorio, 'modelo_arbol.pkl'))