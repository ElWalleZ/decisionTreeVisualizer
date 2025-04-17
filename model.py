import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score


def entrenar_arbol_decision(dataset, max_depth, test_size=0.3):
    """
        Entrena un árbol de decisión (clasificación o regresión) y devuelve métricas y representación del modelo.

        Detecta automáticamente si el problema es de clasificación o regresión basado en los datos objetivo.
        Para clasificación usa DecisionTreeClassifier y accuracy_score, para regresión usa DecisionTreeRegressor y R².

        Args:
            dataset (Bunch): Objeto dataset con:
                - data: array de características
                - target: array de valores objetivo
                - feature_names: lista de nombres de características
                - target_names: lista de nombres de clases (para clasificación)
            max_depth (int): Profundidad máxima del árbol de decisión.
            test_size (float, optional): Proporción del dataset a usar como test (0-1). Default 0.3.

        Returns:
            dict: Diccionario con:
                - 'modelo': Modelo entrenado (DecisionTreeClassifier o DecisionTreeRegressor)
                - 'texto_arbol': Representación textual del árbol (str)
                - 'metrica': Valor de la métrica de evaluación (float)
                - 'tipo': Tipo de problema ('clasificacion' o 'regresion')
                - 'metric_name': Nombre de la métrica usada ('Precisión' o 'R²')

        Notas:
            - Para clasificación, si target_names no coincide con las clases únicas,
              se generan nombres automáticamente.
            - Usa random_state=42 para reproducibilidad en el train-test split.
        """
    X, y = dataset.data, dataset.target
    es_clasificacion = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 10
    Model = DecisionTreeClassifier if es_clasificacion else DecisionTreeRegressor
    modelo = Model(max_depth=max_depth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metrica = accuracy_score(y_test, y_pred) if es_clasificacion else r2_score(y_test, y_pred)

    fnames = getattr(dataset, 'feature_names', [f"X{i}" for i in range(X.shape[1])])
    export_kwargs = dict(feature_names=fnames)
    if es_clasificacion:
        unique_classes = np.unique(y)
        if len(dataset.target_names) != len(unique_classes):
            export_kwargs['class_names'] = [str(cls) for cls in unique_classes]
        else:
            export_kwargs['class_names'] = dataset.target_names
    texto_arbol = export_text(modelo, **export_kwargs)

    return {
        'modelo': modelo,
        'texto_arbol': texto_arbol,
        'metrica': metrica,
        'tipo': 'clasificacion' if es_clasificacion else 'regresion',
        'metric_name': 'Precisión' if es_clasificacion else 'R²'
    }