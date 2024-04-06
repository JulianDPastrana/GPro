import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    

def streamflow_dataset():
    df = pd.read_excel(
        io="./useful_volume.xlsx",
        header=0,
        index_col=0,
        parse_dates=True,
    )
    df = df.clip(lower=0)
    # print(df.describe().T)
    return df[["AGREGADO BOGOTA", "CALIMA1", "MIRAFLORES", "PENOL",
               "PLAYAS", "PUNCHINA", "BETANIA", "CHUZA",
               "ESMERALDA", "GUAVIO", "PRADO", "RIOGRANDE2",
               "SAN LORENZO", "TRONERAS", "URRA1", "SALVAJINA"
               ]
               ]

def thermo_dataset():
    df = pd.read_excel(
        io="./termo_gen.xlsx",
        header=0,
        index_col=0,
        parse_dates=True,
    )
    df = df.clip(lower=0)
    # print(df.describe().T)
    return df[["Total [Gwh]"]]

def get_uv_data():
    df = thermo_dataset()
    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[df.columns] = scaler.fit_transform(df)
    # print(df_norm.describe().T)
    # print(df_norm.info())

    window = WindowGenerator(1, 1, 1, df_norm.columns)

    X, Y = window.make_dataset(df_norm)

    # Find rows with NaNs in X and Y to ensure continuity
    nan_rows_X = np.any(np.isnan(X), axis=1)
    nan_rows_Y = np.any(np.isnan(Y), axis=1)
    nan_rows = nan_rows_X | nan_rows_Y   
    # Filter out the rows with NaNs
    X_clean, Y_clean = X[~nan_rows,:], Y[~nan_rows,:]
    Y_clean = np.maximum(Y_clean, 1e-3)
    # Split into train and test sets
    N = Y_clean.shape[0]
    train_data = (X_clean[0:int(N*0.9)], Y_clean[0:int(N*0.9)])
    val_data = (X_clean[int(N*0.9):int(N*0.95)], Y_clean[int(N*0.9):int(N*0.95)])
    test_data = (X_clean[int(N*0.95):], Y_clean[int(N*0.95):])
    return train_data, val_data, test_data

def main():
    train_data, val_data, test_data = get_uv_data()
    X_train, Y_train = train_data

    X_val, Y_val = val_data
    X_test, Y_test = test_data
    print(np.sum(~np.isfinite(Y_train)))
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)
    # X_test, Y_test = S_test[0]
    plt.plot(Y_test[:, 0])
    plt.show()

if __name__ == "__main__":
    main()
