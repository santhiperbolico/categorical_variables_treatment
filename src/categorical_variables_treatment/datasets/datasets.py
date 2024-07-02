from abc import ABC, abstractmethod

import pandas as pd
from attr import attrs
from sklearn.preprocessing import LabelEncoder

from categorical_variables_treatment.models.models import ModelsType


@attrs
class DataSet(ABC):
    name: str = ""
    url: str = ''
    model_type: ModelsType = ModelsType.Regressor
    categorical_features: list[str] = []

    @abstractmethod
    def get_dataset(self) -> pd.DataFrame:
        pass


@attrs
class HousingDataSet(DataSet):
    """
    Función que descarga y formatea el dataset, devolciendo un DataFrame.

    Returns
    -------
    df: pd.DataFrame
        DataFrame con el dataset.
    """

    name = "housing"
    url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
    model_type = ModelsType.Regressor
    categorical_features = ["ORGANIZATION_TYPE"]

    def get_dataset(self) -> pd.DataFrame:
        # Cargar el conjunto de datos (disponible en Kaggle:
        df = pd.read_csv(self.url)
        # Renombrar las columnas para claridad en el ejemplo
        df = df.rename(columns=dict(
            zip(['ocean_proximity', 'median_house_value'], ['ORGANIZATION_TYPE', 'TARGET'])))
        # Eliminamos las variables de longitud y latitude
        df = df.drop(columns=["longitude", "latitude"])
        # Eliminar aquellas filas con algún dato nulo.
        df = df[~df.isna().any(axis=1)]
        return df


@attrs
class VideoGamesDataSet(DataSet):

    name = "video_games_sales"
    url = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/video_games_sales.csv"
    model_type = ModelsType.Regressor
    categorical_features = ["Platform", "years", "Genre", "Publisher"]

    def get_dataset(self) -> pd.DataFrame:
        """
        Función que descarga y formatea el dataset, devolciendo un DataFrame.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con el dataset.
        """
        df = pd.read_csv(self.url)
        columns = ["Name", "Platform", "Year_of_Release", "Genre", "Publisher", "Global_Sales",
                   "User_Score"]
        df = df[columns]
        df = df[~df.isna().any(axis=1)]
        (df["User_Score"] == "tbd").sum()
        publishers = df.Publisher.value_counts()
        others_publishers = publishers[publishers <= 150]
        df.loc[df.Publisher.isin(others_publishers.index), "Publisher"] = "Other"
        df.Year_of_Release = df.Year_of_Release.astype(int)
        df["years"] = df.Year_of_Release.apply(lambda x: f"dec_{x - x % 10}")
        df = df.rename(columns={"User_Score": "TARGET"})
        df = df[["Platform", "years", "Genre", "Publisher", "Global_Sales", "TARGET"]]
        df = df[~(df["TARGET"] == "tbd")]
        df.TARGET = df.TARGET.astype(float) / 10
        return df


@attrs
class AdultsDataSet(DataSet):

    name = "adults"
    url = "https://raw.githubusercontent.com/selva86/datasets/master/adultTrain.csv"
    model_type = ModelsType.Classifier
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation',
                        'relationship', 'race', 'sex', 'native_country']

    def get_dataset(self) -> pd.DataFrame:
        """
        Función que descarga y formatea el dataset, devolciendo un DataFrame.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con el dataset.
        """
        df = pd.read_csv(self.url)
        df = df.rename(columns={"class": "TARGET"}).drop(columns=["fnlwgt", "education_num"])
        label_encoder = LabelEncoder()
        df["TARGET"] = label_encoder.fit_transform(df["TARGET"])
        return df


def get_dataset(name_dataset: str) -> DataSet:
    """
    Función que devuelve el dataset indicado

    Parameters
    ----------
    name_dataset: str
        Nombre del dataset

    Returns
    -------
    dataset: DataSet
        Objeto dataset seleccionado.
    """

    dic_datasets = {
        VideoGamesDataSet.name: VideoGamesDataSet,
        AdultsDataSet.name: AdultsDataSet,
        HousingDataSet.name: HousingDataSet
    }
    try:
        return dic_datasets[name_dataset]()
    except KeyError:
        raise ValueError(f"El dataset {name_dataset} no está implementado. Prueba con"
                         f"alguno de: {list(dic_datasets.keys())}")

