#!/home/usuario/Documents/GPro/mygpvenv/bin/python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class WindowGenerator:
    """
    A class for generating input and label windows for time series data.

    Attributes:
        input_width (int): The width of the input window.
        label_width (int): The width of the label window.
        shift (int): The shift between the input and label windows.
        label_columns (list): List of label column names.
    """

    def __init__(self, input_width: int, label_width: int, shift: int, label_columns: list = None) -> None:
        """
        Initializes the WindowGenerator with the specified parameters.

        Args:
            input_width (int): The width of the input window.
            label_width (int): The width of the label window.
            shift (int): The shift between the input and label windows.
            label_columns (list, optional): List of label column names. Defaults to None.
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_columns = label_columns

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, self.total_window_size)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, index: int = 0) -> tuple:
        """
        Splits the window into input and label windows.

        Args:
            index (int): The starting index for the split. Defaults to 0.

        Returns:
            tuple: Tuple containing input and label windows.
        """
        features = self.data
        inputs = features[index + self.input_indices]
        labels = features[index + self.label_indices]

        if self.label_columns is not None:
            labels = np.stack([labels[:, self.label_columns_indices[name]] for name in self.label_columns], axis=-1)

        return inputs, labels

    def make_dataset(self, dataframe: pd.DataFrame) -> tuple:
        """
        Creates the dataset from the input dataframe.

        Args:
            dataframe (pd.DataFrame): Input data.

        Returns:
            tuple: Tuple containing input and label datasets.
        """
        self.label_columns_indices = {name: j for j, name in enumerate(dataframe.columns)}
        self.data = dataframe.values
        n_samples, n_feats = dataframe.shape
        real_samples = n_samples - self.total_window_size + 1

        x_data = np.empty((real_samples, n_feats * self.input_width))
        y_data = np.empty((real_samples, len(self.label_columns) * self.label_width))

        for index in range(real_samples):
            inputs, labels = self.split_window(index)
            x_data[index] = inputs.ravel()
            y_data[index] = labels.ravel()

        return x_data, y_data



def get_daily_vol_data(input_width: int = 1, label_width: int = 1, shift: int = 1) -> tuple:
    """
    Loads, preprocesses, scales the streamflow dataset, and splits it into 
    train and test sets.

    Returns:
        tuple: Tuple containing train and test sets.
    """
    # Load and preprocess the dataset
    df = pd.read_excel(
        io="./PorcVoluUtilDiar.xlsx",
        header=0,
        index_col=0,
        parse_dates=True,
    )
    eps = 1e-6
    df = df.clip(lower=eps, upper=1 - eps)
    df = df[
        ["AGREGADO BOGOTA", "CALIMA1", "MIRAFLORES", "PENOL", "PLAYAS", "PUNCHINA", 
         "BETANIA", "CHUZA", "ESMERALDA", "GUAVIO", "PRADO", "RIOGRANDE2", "SAN LORENZO", 
         "TRONERAS", "URRA1", "SALVAJINA"]
    ]

    # Rename columns to single characters
    df.columns = [chr(65 + i) for i in range(len(df.columns))]

    # Generate dataset windows
    window = WindowGenerator(input_width, label_width, shift, label_columns=df.columns)
    X, Y = window.make_dataset(df)

    # Remove rows with NaNs
    nan_rows = np.any(np.isnan(X), axis=1) | np.any(np.isnan(Y), axis=1)
    X_clean, Y_clean = X[~nan_rows], Y[~nan_rows]

    # Split into training and testing sets
    N = Y_clean.shape[0]
    tst = 446
    tr = N - tst

    train_data = (X_clean[:tr], Y_clean[:tr])
    test_data = (X_clean[tr:], Y_clean[tr:])

    return train_data, test_data



def main():
    """
    Main function to prepare the dataset and split into train, validation, and test sets.
    """
    train_data, test_data = get_daily_vol_data()
    X_train, Y_train = train_data
    X_test, Y_test = test_data
    print(f"X train size: {X_train.shape}, Y train size: {Y_train.shape}")
    print(f"X test size: {X_test.shape}, Y train size: {Y_test.shape}")


if __name__ == "__main__":
    main()
