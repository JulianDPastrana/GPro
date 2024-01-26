import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# class WindowGenerator:
#     def __init__(
#         self,
#         input_width,
#         label_width,
#         shift,
#         train_df,
#         val_df,
#         test_df,
#         label_columns=None,
#         input_columns=None,
#     ):
#         # Store the raw data.
#         self.train_df = train_df
#         self.val_df = val_df
#         self.test_df = test_df

#         # Work out the label column indices.
#         self.label_columns = label_columns
#         self.input_columns = input_columns
#         if label_columns is not None:
#             self.label_columns_indices = {
#                 name: i for i, name in enumerate(label_columns)
#             }
#         self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

#         # Work out the window parameters.
#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift

#         self.total_window_size = input_width + shift

#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]

#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

#     def __repr__(self):
#         return "\n".join(
#             [
#                 f"Total window size: {self.total_window_size}",
#                 f"Input indices: {self.input_indices}",
#                 f"Label indices: {self.label_indices}",
#                 f"Label column name(s): {self.label_columns}",
#                 f"Input column name(s): {self.input_columns}",
#             ]
#         )


# def split_window(self, features):
#     inputs = features[:, self.input_slice, :]
#     labels = features[:, self.labels_slice, :]
#     if self.label_columns is not None:
#         labels = tf.stack(
#             [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#             axis=-1,
#         )

#     if self.input_columns is not None:
#         inputs = tf.stack(
#             [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
#             axis=-1,
#         )

#     # Slicing doesn't preserve static shape information, so set the shapes
#     # manually. This way the `tf.data.Datasets` are easier to inspect.
#     inputs.set_shape([None, self.input_width, None])
#     labels.set_shape([None, self.label_width, None])

#     return inputs, labels


# WindowGenerator.split_window = split_window


# def make_dataset(self, data):
#     data = np.array(data, dtype=np.float64)
#     ds = tf.keras.utils.timeseries_dataset_from_array(
#         data=data,
#         targets=None,
#         sequence_length=self.total_window_size,
#         sequence_stride=1,
#         shuffle=False,
#         batch_size=256,
#     )

#     ds = ds.map(self.split_window)

#     return ds


# WindowGenerator.make_dataset = make_dataset


# @property
# def train(self):
#     return self.make_dataset(self.train_df)


# @property
# def val(self):
#     return self.make_dataset(self.val_df)


# @property
# def test(self):
#     return self.make_dataset(self.test_df)


# @property
# def example(self):
#     """Get and cache an example batch of `inputs, labels` for plotting."""
#     result = getattr(self, "_example", None)
#     if result is None:
#         # No example batch was found, so get one from the `.train` dataset
#         result = next(iter(self.train))
#         # And cache it for next time
#         self._example = result
#     return result


# WindowGenerator.train = train
# WindowGenerator.val = val
# WindowGenerator.test = test
# WindowGenerator.example = example

# def streamflow_dataset(
#         input_width,
#         label_width,
#         shift
# ):
#     df = pd.read_excel(
#         io='./streamflow_dataset.xlsx',
#         header=0,
#         index_col=0
#     )

#     column_indices = {name: i for i, name in enumerate(df.columns)}

#     n = len(df)
#     train_df = df[0:int(n*0.7)]
#     val_df = df[int(n*0.7):int(n*0.9)]
#     test_df = df[int(n*0.9):]

#     num_features = df.shape[1]

#     train_mean = train_df.mean()
#     train_std = train_df.std()

#     for data_df in [train_df, val_df, test_df]:
#         data_df = (data_df - train_mean) / train_std
#         data_df.fillna(0, inplace=True)

#     data_windowed = WindowGenerator(
#         input_width,
#         label_width,
#         shift,
#         train_df,
#         val_df,
#         test_df,
#         label_columns=df.columns,
#         input_columns=df.columns,
#     )

#     return data_windowed



class WindowGenerator:
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 label_columns: list = None) -> None:

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

    def split_window(self,
                     index: int = 0) -> tuple:

        features = self.data
        inputs = features[index + self.input_indices]
        labels = features[index + self.label_indices]

        if self.label_columns is not None:
            labels = np.stack(
                [labels[:, self.label_columns_indices[name]] for name in self.label_columns],
                axis=-1
            )

        return inputs, labels

    def make_dataset(self, dataframe: pd.DataFrame) -> tuple:

        self.label_columns_indices = {name: j for j, name in
                                      enumerate(dataframe.columns)}
        self.data = dataframe.values
        n_samples, n_feats = dataframe.shape
        real_samples = n_samples - self.total_window_size + 1

        x_data = np.empty(
            shape=(real_samples, n_feats * self.input_width)
        )
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
        return ("\033[1mWindow Information\033[0m\n"
                f"Total window size: {self.total_window_size}\n"
                f"Input indices: {self.input_indices}\n"
                f"Label indices: {self.label_indices}\n"
                f"Label column name(s): {label_columns}\n")



def toy_datset(
        input_width=1,
        label_width=1,
        shift=1,
        N=1001,
        verbose=True,
        split_data=0.8
):


    # Build inputs X
    X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

    # Deterministic functions in place of latent ones
    f11 = np.sin
    f12 = np.cos

    f21 = np.cos
    f22 = np.cos

    f31 = lambda x: np.cos(x) * np.sin(x)
    f32 = lambda x: 0.1 * x

    # Compute loc and scale as functions of input X
    loc1 = f11(X)
    scale1 = np.heaviside(f12(X), 1)

    loc2 = f21(X)
    scale2 = np.exp(f22(X))

    loc3 = f31(X)
    scale3 = np.exp(f32(X))

    # Sample outputs Y from Gaussian Likelihood
    Y = np.concatenate(
        [
            np.random.normal(loc1, scale1),
            np.random.normal(loc2, scale2),
            np.random.normal(loc3, scale3)
        ], axis=-1
    )

    df = pd.DataFrame(
        data = Y,
        columns = [f"Task {i}" for i in range(1, 4)]
    )
    window = WindowGenerator(
        input_width,
        label_width,
        shift,
        df.columns
    )
    

    X, Y = window.make_dataset(df)
    threshold = int(N*split_data)
    train_data = X[0:threshold], Y[0:threshold]
    test_data = X[threshold:], Y[threshold:]

    if verbose:
        print(window)
        print(f"Num train: {threshold}\n"
              f"Num test: {N - threshold}\n")

    return train_data, test_data


def lognorm_datset(
        input_width=1,
        label_width=1,
        shift=1,
        N=1001,
        verbose=True,
        split_data=0.8
):


    # Build inputs X
    X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

    # Deterministic functions in place of latent ones
    f1 = np.sin
    f2 = np.cos

    # Compute loc and scale as functions of input X
    loc = f1(X)
    scale = np.exp(f2(X))

    # Sample outputs Y from Gaussian Likelihood
    Y = np.random.lognormal(loc, scale)

    df = pd.DataFrame(
        data = Y,
        columns = [f"LogNorm Task"]
    )
    window = WindowGenerator(
        input_width,
        label_width,
        shift,
        df.columns
    )
    

    # X, Y = window.make_dataset(df)
    threshold = int(N*split_data)
    train_data = X[0:threshold], Y[0:threshold]
    test_data = X[threshold:], Y[threshold:]

    if verbose:
        print(window)
        print(f"Num train: {threshold}\n"
              f"Num test: {N - threshold}\n")

    return train_data, test_data


def main():
    lognorm_datset()

if __name__=="__main__":
    main()