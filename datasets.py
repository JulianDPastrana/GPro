import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def streamflow_dataset(
    input_width=1, label_width=1, shift=1, verbose=True, split_data=0.8
):
    df = pd.read_excel(io="./streamflow_dataset.xlsx", header=0, index_col=0)
    df = df.iloc[:, 0].to_frame()
    df = df / df.max()
    df.fillna(0, inplace=True)
    print(df.describe())
    column_indices = {name: i for i, name in enumerate(df.columns)}

    N = len(df)
    window = WindowGenerator(input_width, label_width, shift, df.columns)

    X, Y = window.make_dataset(df)
    threshold = int(N * split_data)
    train_data = X[0:threshold], Y[0:threshold]
    test_data = X[threshold:], Y[threshold:]

    if verbose:
        print(window)
        print(f"Num train: {threshold}\n" f"Num test: {N - threshold}\n")

    return train_data, test_data


def toy_datset(
    N=2001, verbose=True, split_data=0.8
):
    # Build inputs X
    X = np.linspace(0, 8 * np.pi, N)[:, None]  # X must be of shape [N, 1]

    # Deterministic functions in place of latent ones
    f1 = np.sin
    f2 = np.cos


    # Compute loc and scale as functions of input X
    loc = f1(X)
    scale = np.exp(0.1*f2(X))

    # Sample outputs Y from Gaussian Likelihood
    Y = np.exp(np.random.normal(loc, scale))

    threshold = int(N * split_data)
    train_data = X[0:threshold], Y[0:threshold]
    test_data = X[threshold:], Y[threshold:]

    if verbose:
        print(f"Num train: {threshold}\n" f"Num test: {N - threshold}\n")

    return train_data, test_data


def simple_datset(
    input_width=1, label_width=1, shift=1, N=1001, verbose=True, split_data=0.8
):
    # Build inputs X
    X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

    # Deterministic functions in place of latent ones
    f1 = np.cos
    f2 = np.sin

    # Compute loc and scale as functions of input X
    loc = f1(X)
    beta = np.exp(f2(X))

    # Sample outputs Y from Gaussian Likelihood
    Y = 0.5 * np.cos(X) + 1 + np.random.normal(0, 0.1, size=(N, 1))

    df = pd.DataFrame(data=Y, columns=[f"LogNorm Task"])
    window = WindowGenerator(input_width, label_width, shift, df.columns)

    # X, Y = window.make_dataset(df)
    threshold = int(N * split_data)
    train_data = X[0:threshold], Y[0:threshold]
    test_data = X[threshold:], Y[threshold:]

    if verbose:
        print(window)
        print(f"Num train: {threshold}\n" f"Num test: {N - threshold}\n")

    return train_data, test_data


def main():
    train_data, test_data = toy_datset()
    x_train, y_train = train_data
    plt.scatter(range(len(y_train)), y_train)
    plt.show()


if __name__ == "__main__":
    main()
