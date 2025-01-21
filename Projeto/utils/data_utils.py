import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """Carrega dados de um arquivo CSV"""
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    """Realiza o pré-processamento dos dados."""
    # Exemplo de pré-processamento: remover valores ausentes
    data = data.dropna()
    return data


def split_data(data, target_column):
    """Separa os dados em conjuntos de treinamento e teste."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Escala os recursos usando StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def encode_features(data, features, encoding_type='label', drop_original=False):
    """
    Codifica as variáveis categoricas do DataFlame

    Args:
        data (DataFrame): The input DataFrame containing the features to encode.
        features (list): A list of column names to encode.
        encoding_type (str): The type of encoding to apply. Options:
            - 'label': Applies Label Encoding.
            - 'onehot': Applies One-Hot Encoding.
        drop_original (bool): If True, drops the original columns after encoding (only for 'onehot').

    Returns:
        DataFrame: The DataFrame with encoded features.

    Raises:
        ValueError: If an invalid encoding_type is provided or features are not in the DataFrame.

    """
    if encoding_type not in ['label', 'onehot']:
        raise ValueError("Invalid encoding_type. Choose 'label' or 'onehot'.")

    try:
        encoded_data = data.copy()

        if encoding_type == 'label':
            le = LabelEncoder()
            for feature in features:
                if feature not in data.columns:
                    raise ValueError(f"Feature '{feature}' not found in DataFrame.")
                encoded_data[feature] = le.fit_transform(data[feature])

        elif encoding_type == 'onehot':
            for feature in features:
                if feature not in data.columns:
                    raise ValueError(f"Feature '{feature}' not found in DataFrame.")
                onehot = pd.get_dummies(data[feature], prefix=feature)
                encoded_data = pd.concat([encoded_data, onehot], axis=1)
                if drop_original:
                    encoded_data.drop(columns=[feature], inplace=True)

        return encoded_data

    except Exception as e:
        raise Exception(f"Error during encoding: {e}")
