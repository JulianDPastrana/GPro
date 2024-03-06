import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


class WindowGenerator:
    def __init__(
        self, input_width: int, label_width: int, shift: int, label_columns: list = None
    ) -> None:
        self.data = None
        self.label_columns_indices = None
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
        features = self.data
        inputs = features[index + self.input_indices]
        labels = features[index + self.label_indices]

        if self.label_columns is not None:
            labels = np.stack(
                [
                    labels[:, self.label_columns_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        return inputs, labels

    def make_dataset(self, dataframe: pd.DataFrame) -> tuple:
        self.label_columns_indices = {
            name: j for j, name in enumerate(dataframe.columns)
        }
        self.data = dataframe.values
        n_samples, n_feats = dataframe.shape
        real_samples = n_samples - self.total_window_size + 1

        x_data = np.empty(shape=(real_samples, n_feats * self.input_width))
        y_data = np.empty(
            shape=(real_samples, self.label_columns.__len__() * self.label_width)
        )

        for index in range(real_samples):
            inputs, labels = self.split_window(index=index)
            x_data[index] = inputs.ravel()
            y_data[index] = labels.ravel()

        return x_data, y_data

    def __repr__(self):
        label_columns = ", ".join(str(label) for label in self.label_columns)
        return (
            "\033[1mWindow Information\033[0m\n"
            f"Total window size: {self.total_window_size}\n"
            f"Input indices: {self.input_indices}\n"
            f"Label indices: {self.label_indices}\n"
            f"Label column name(s): {label_columns}\n"
        )


def streamflow_dataset():
    df = pd.read_excel(
        io="./useful_volume.xlsx",
        header=0,
        index_col=0,
    )
    return df.iloc[:, 1:2]




def get_uv_data():
    df = streamflow_dataset()
    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[df.columns] = scaler.fit_transform(df)
    print(df_norm.describe().T)

    window = WindowGenerator(1, 1, 1, df_norm.columns)

    X, Y = window.make_dataset(df_norm)
    # Find rows with NaNs in X and Y
    nan_rows_X = np.any(np.isnan(X), axis=1)
    nan_rows_Y = np.any(np.isnan(Y), axis=1)

    # Combine the conditions to find rows with NaNs in either X or Y
    nan_rows = nan_rows_X | nan_rows_Y

    # Select rows where there are no NaNs in either X or Y
    X = X[~nan_rows]
    Y = Y[~nan_rows]
    assert X.shape[0] == Y.shape[0]
    N = Y.shape[0]
    split_data = 0.8
    threshold = int(N * split_data)
    train_data = X[0:threshold], Y[0:threshold]
    test_data = X[threshold:], Y[threshold:]
    print(window)
    print(f"Num train: {threshold}\n" f"Num test: {N - threshold}\n")
    return train_data, test_data


def main():

    train_data, test_data = get_uv_data()
    X_test, Y_test = test_data
    X_train, Y_train = train_data
    plt.plot(Y_train)
    plt.show()
    print(np.isnan(X_train).sum(), np.isnan(Y_train).sum())
    print(X_train.shape, Y_train.shape)
    index = np.concatenate((np.argwhere(~np.isnan(X_train)), np.argwhere(~np.isnan(Y_train))), axis=0)
    print(index.shape)
    print(index)




if __name__ == "__main__":
    main()