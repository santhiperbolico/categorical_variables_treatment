from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import lazypredict
import xgboost
import lightgbm
import sklearn
from lazypredict.Supervised import LazyRegressor, LazyClassifier


lazypredict.Supervised.REGRESSORS = [
    ('XGBRegressor', xgboost.sklearn.XGBRegressor),
    ('LGBMRegressor', lightgbm.sklearn.LGBMRegressor),
    ('HistGradientBoostingRegressor',
  sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor),
    ('RandomForestRegressor', sklearn.ensemble._forest.RandomForestRegressor),
     ('LinearRegression', sklearn.linear_model._base.LinearRegression),
    ('BayesianRidge', sklearn.linear_model._bayes.BayesianRidge),

]

lazypredict.Supervised.CLASSIFIERS = [
    ('XGBClassifier', xgboost.sklearn.XGBClassifier),
    ('LGBMClassifier', lightgbm.sklearn.LGBMClassifier),
    ('StackingClassifier', sklearn.ensemble._stacking.StackingClassifier),
    ('RandomForestClassifier', sklearn.ensemble._forest.RandomForestClassifier),
    ('KNeighborsClassifier',
  sklearn.neighbors._classification.KNeighborsClassifier),
    ('LogisticRegression', sklearn.linear_model._logistic.LogisticRegression),
    ('BernoulliNB', sklearn.naive_bayes.BernoulliNB),
]


LazyModels = Union[LazyRegressor, LazyClassifier]


class ModelsType(str, Enum):
    Regressor = "regressor"
    Classifier = "classifier"


def get_lazy_model(model_type: ModelsType) -> LazyModels:
    """
    Función que devuleve instanciado un tipo de modelo de LazyPredict, según lo indicado
    port model_type.
    
    Parameters
    ----------
    model_type: ModelsType
        Tipo de modelo

    Returns
    -------
    lazymodel: LazyModels
        Modelo de LazyPredict instanciado.
    """
    dic_models = {
      ModelsType.Regressor: LazyRegressor(
          verbose=0,
          ignore_warnings=False
          ),
      ModelsType.Classifier: LazyClassifier(verbose=0,
          ignore_warnings=False)
    }
    try:
        return dic_models[model_type]
    except KeyError:
        raise ValueError(f"No está implementado el tipo de modelo {model_type}. Prueba"
                     f"con alguno: {list(dic_models.keys())}")


def evaluate_dataset(
        df: pd.DataFrame,
        sample_size: int = None,
        model_type: ModelsType = ModelsType.Regressor
) -> pd.DataFrame:
    """
    Función que evalua usando lazypredict el dataset df
    Parameters
    ----------
    df: pd.DataFrame
        Tabla con los datos
    sample_size: int
        Tamaño de la muestra que se quiere usar, si no se indica usa todo.
    model_type: ModelsType
        Tipo de modelo.

    Returns
    -------
    models: pd.DataFrame
        Tabla con los resultados de los modelos probados.
    """
    # Seleccionar una muestra aleatoria de 2000 filas
    # Dividir el conjunto de datos en entrenamiento y prueba
    np.random.seed(123456789)
    df_models = df.copy()
    if sample_size:
        df_models = df.sample(sample_size)
    X = df_models.drop(columns=["TARGET"])
    y = df_models['TARGET'].values
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de regresión lineal
    cols_to_scaler = [col for col in X_train.columns if not "embed_" in col]
    X_train[cols_to_scaler] = scaler.fit_transform(X_train[cols_to_scaler])
    X_test[cols_to_scaler] = scaler.transform(X_test[cols_to_scaler])
    reg = get_lazy_model(model_type)
    models,predictions = reg.fit(X_train, X_test, y_train, y_test)
    return models
