import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from attr import attrs
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from openai import OpenAI

from categorical_variables_treatment.datasets.datasets import DataSet
from categorical_variables_treatment.models.models import ModelsType


def create_embedding(client: OpenAI, values: np.ndarray, col_name: str) -> pd.DataFrame:
    """
    Función que crea un embeding usando el modelo de openai

    Parameters
    ----------
    client: OpenAI
        Cliente de OpenAI
    values: np.ndarray
        Valores que pasar por el embedding.
    col_name: str
        Nombre de la columna que se quiere aplicar el embedding.

    Returns
    -------
    df_data: pd.DataFrame
        Tabla de datos con el embedding

    """
    embedding_list = []
    columns = []
    k = 0
    for val in values:
        response = client.embeddings.create(
          input=f"{col_name}: {val}",
          model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        print(f" Embedding {val}")
        print(f"\t -Data Len {len(response.data)}")
        print(f"\t -Embedding Shape {len(embedding)}")
        embedding_list.append(embedding)
        columns.append(f"{col_name}_embed_{k}")
        k += 1

    df_data = pd.DataFrame(np.array(embedding_list), index=values)
    return df_data


def get_new_base(df_data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Función que reduce el tamaño del embeding de df_data.

    Parameters
    ----------
    df_data: pd.DataFrame
        Tabla de datos con el embedding
    col_name: str
        Nombre de la columna procesada.

    Returns
    -------
    pca_embeding: pd.DataFrame
        tabla de datos con el embedding reducido.
    """
    pca = PCA(n_components=np.linalg.matrix_rank(df_data))
    pca_embedding = pd.DataFrame(pca.fit_transform(df_data), index=df_data.index)
    pca_embedding.columns = [f"{col_name}_embed_{i}" for i in pca_embedding.columns]
    pca_embedding.index.name = col_name
    return pca_embedding


def generate_embedding_model(
    df_data: pd.DataFrame,
    categorical_features: list[str],
    size_embedding: list[int] = None,
    model_type: ModelsType = ModelsType.Regressor
    ) -> tuple[tf.keras.models.Model, tf.keras.models.Model]:
    """
    Función que genera una red neuronal que aplica embedding a las columnas etiquetadas como
    categoricas.

    Parameters
    ----------
    df_data: pd.DataFrame,
        Tabla de datos
    categorical_features: list[str]
        Lista de nombres de las las variables categoricas.
    size_embedding: list[int], default None
        Tamaño de las capas finales del embedding, si es None se toma como tamaño el
        número de valores únicos.
    model_type: ModelsType, default ModelsType.Regressor
        Tipo de modelo.

    Returns
    -------
    model_full: tf.keras.models.Model
        Red neuronal completa
    model_embedding: tf.keras.models.Model
        Parte del embedding de la red neuronal.
    """

    np.random.seed(1234)
    models = []
    inputs = []

    activation_output = "relu"
    loss_function = "mse"
    if model_type == "classifier":
        activation_output = "sigmoid"
        loss_function = "binary_crossentropy"

    if size_embedding is None:
        size_embedding = [None] * len(categorical_features)

    if isinstance(size_embedding, int):
        size_embedding = [size_embedding] * len(categorical_features)

    # Crea los embedding para cada variable categorica
    for k, cat in enumerate(categorical_features):
        n_features = df_data[cat].unique().size
        n_embedding = n_features
        if size_embedding[k]:
            n_embedding = size_embedding[k]

        vocab_size = df_data[cat].nunique()
        inpt = tf.keras.layers.Input(shape=(1,), name='input_' + cat)
        embed = tf.keras.layers.Embedding(vocab_size, n_embedding, trainable=True,
                                        embeddings_initializer=tf.initializers.random_normal)(inpt)
        embed_reshaped = tf.keras.layers.Reshape(target_shape=(n_embedding,))(embed)
        if n_embedding > n_features:
            embed_reshaped = tf.keras.layers.Dense(n_features)(embed_reshaped)
        models.append(embed_reshaped)
        inputs.append(inpt)

    merge_models = tf.keras.layers.concatenate(models)
    pre_preds = tf.keras.layers.Dense(1000)(merge_models)
    pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
    pre_preds = tf.keras.layers.Dense(500)(pre_preds)
    pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
    pred = tf.keras.layers.Dense(1, activation=activation_output)(pre_preds)

    model_embedding = tf.keras.models.Model(inputs=inputs, outputs=merge_models)
    model_full = tf.keras.models.Model(inputs=inputs, outputs=pred)

    model_full.compile(optimizer='adam', loss=loss_function)

    return model_full, model_embedding


@attrs
class Encoding(ABC):

    @abstractmethod
    def execute(self, dataset: DataSet) -> pd.DataFrame:
        pass


@attrs
class OneHotEncoding(Encoding):
    name = "one-hot-encoder"

    def execute(self, dataset: DataSet) -> pd.DataFrame:
        """
        Función que ejecuta la tranformación de las variables categoricas.

        Parameters
        ----------
        dataset: DataSet
            Dataset a preprocesar.

        Returns
        -------
        df_one_hot: pd.DataFrame
            Tabla con las variables categoricas transformadas.
        """
        df = dataset.get_dataset()
        df_one_hot = df.copy().drop(columns=dataset.categorical_features)
        df_dumimes = pd.get_dummies(df[dataset.categorical_features]).astype(int)
        df_one_hot = pd.concat((df_one_hot, df_dumimes), axis=1)
        return df_one_hot


@attrs
class LabelEncoding(Encoding):
    name = "label-encoder"

    def execute(self, dataset: DataSet) -> pd.DataFrame:
        """
        Función que ejecuta la tranformación de las variables categoricas.

        Parameters
        ----------
        dataset: DataSet
            Dataset a preprocesar.

        Returns
        -------
        df_encoder: pd.DataFrame
            Tabla con las variables categoricas transformadas.
        """
        df = dataset.get_dataset()
        label_encoder = LabelEncoder()
        df_encoder = df.copy().drop(columns=dataset.categorical_features)
        for col in dataset.categorical_features:
            df_encoder[f"{col}_COD"] = label_encoder.fit_transform(df[col])
        return df_encoder


@attrs
class EmbeddingEncoding(Encoding):
    name = "embeding-encoder"

    def execute(
            self,
            dataset: DataSet,
            epochs: int = 20,
            batch_size: int = 64,
            verbose: int = 1
    ) -> pd.DataFrame:
        """
        Función que ejecuta la tranformación de las variables categoricas.

        Parameters
        ----------
        dataset: DataSet
            Dataset a preprocesar.
        epochs: int, default 20
            Epochs de la red neuronal que aplica el embeding
        batch_size: int, default 64
            Tamaño de los batch del entrenamiento de la red neuronal.
        verbose: int, default 1
            Verbose del entrenamiento de la red neuronal.

        Returns
        -------
        df_embeding: pd.DataFrame
            Tabla con las variables categoricas transformadas.
        """
        df = dataset.get_dataset()
        label_encoder = LabelEncoder()
        df_embedding = df.copy().drop(columns=dataset.categorical_features)
        categorical_features_cod = []
        for col in dataset.categorical_features:
            df_embedding[f"{col}_COD"] = label_encoder.fit_transform(df[col])
            categorical_features_cod.append(f"{col}_COD")

        # Construimos una red neuronal que construya el embedding
        train_size = int(df_embedding.shape[0] * 0.80)
        df_train = df_embedding.iloc[:train_size, :]
        df_test = df_embedding.iloc[train_size:, :]
        model_full, model_embedding = generate_embedding_model(df_embedding,
                                                               categorical_features_cod,
                                                               model_type=dataset.model_type)
        scaler_x = MinMaxScaler()
        input_num_train = scaler_x.fit_transform(
            df_train.drop(columns=categorical_features_cod + ["TARGET"]))
        input_num_test = scaler_x.transform(
            df_test.drop(columns=categorical_features_cod + ["TARGET"]))

        labels_train = df_train["TARGET"].values
        labels_test = df_test["TARGET"].values

        input_dict_train = {}
        input_dict_test = {}
        input_embedding = {}
        cols_embeddings = []

        for col in categorical_features_cod:
            input_dict_train[f"input_{col}"] = df_train[col].values
            input_dict_test[f"input_{col}"] = df_test[col].values
            input_embedding.update({f"input_{col}": df_embedding[col].values})
            n_features = df_embedding[col].nunique()
            cols_embeddings += [f'{col}_embed_{i}' for i in range(n_features)]

        # Añadir características numéricas
        input_dict_train["input_number_features"] = input_num_train
        input_dict_test["input_number_features"] = input_num_test

        _ = model_full.fit(
            input_dict_train,
            labels_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(input_dict_test, labels_test),
            verbose=1
        )

        # Obtener los embeddings
        embeddings = model_embedding.predict(input_embedding)
        embeddings_df = pd.DataFrame(embeddings, columns=cols_embeddings).reset_index(drop=True)
        df_embedding = df_embedding.drop(columns=categorical_features_cod).reset_index(drop=True)
        df_embedding = pd.concat((df_embedding, embeddings_df), axis=1)
        return df_embedding


@attrs
class LlmEncoding(Encoding):
    name = "llm-encoder"

    def execute(self, dataset: DataSet, api_key: str = None) -> pd.DataFrame:
        """
        Función que ejecuta la tranformación de las variables categoricas.

        Parameters
        ----------
        dataset: DataSet
            Dataset a preprocesar.
        api_key: str
            Api Key de Open Ai, por defecto toma la variable de entorno.

        Returns
        -------
        df_llm_embedding: pd.DataFrame
            Tabla con las variables categoricas transformadas.
        """
        df = dataset.get_dataset()
        params = {"api_key": os.getenv("API_KEY")}
        if api_key:
            params = {"api_key": api_key}
        client = OpenAI(**params)
        df_llm_embedding = df.copy()

        for col in dataset.categorical_features:
            df_col = create_embedding(client, df[col].unique(), col_name=col)
            pca_embedding = get_new_base(df_col, col)
            df_llm_embedding = pd.merge(df, pca_embedding.reset_index(), on=col)

        df_llm_embedding = df_llm_embedding.drop(columns=dataset.categorical_features)
        return df_llm_embedding
