import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score


def entrenar_arbol_decision(dataset, max_depth, test_size=0.3):
    """
    Entrena un árbol de decisión (clasificación o regresión) y devuelve:
      - modelo entrenado
      - texto del árbol
      - métrica de evaluación (accuracy o R²)
      - tipo de problema
    Parámetros:
      dataset: objeto con .data, .target, opcionalmente .feature_names y .target_names
      max_depth: profundidad máxima del árbol
      test_size: proporción para test/train split (si cv is None)
    """
    X, y = dataset.data, dataset.target
    # Detección de tipo
    es_clasificacion = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 10
    Model = DecisionTreeClassifier if es_clasificacion else DecisionTreeRegressor
    modelo = Model(max_depth=max_depth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metrica = accuracy_score(y_test, y_pred) if es_clasificacion else r2_score(y_test, y_pred)

    # Exportar árbol
    fnames = getattr(dataset, 'feature_names', [f"X{i}" for i in range(X.shape[1])])
    export_kwargs = dict(feature_names=fnames)
    if es_clasificacion:
        export_kwargs['class_names'] = getattr(dataset, 'target_names', [str(c) for c in modelo.classes_])
    texto_arbol = export_text(modelo, **export_kwargs)

    return {
        'modelo': modelo,
        'texto_arbol': texto_arbol,
        'metrica': metrica,
        'tipo': 'clasificacion' if es_clasificacion else 'regresion',
        'metric_name': 'Precisión' if es_clasificacion else 'R²'
    }