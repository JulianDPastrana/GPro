import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class WindowGenerator:
    def __init__(self, input_width: int, label_width: int, shift: int, label_columns: list = None) -> None:
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.label_columns = label_columns
        self.total_window_size = input_width + shift
        self.label_columns_indices = None

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features: np.ndarray) -> tuple:
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            labels = np.stack([labels[:, :, self.label_columns_indices[name]] for name in self.label_columns], axis=-1)

        return inputs, labels

    def make_dataset(self, dataframe: pd.DataFrame) -> tuple:
        self.label_columns_indices = {name: i for i, name in enumerate(dataframe.columns) if name in self.label_columns}

        data = dataframe.values
        n_samples = data.shape[0]
        inputs, labels = [], []

        for i in range(n_samples - self.total_window_size + 1):
            window = data[i: i + self.total_window_size, :]
            split_input, split_label = self.split_window(window[np.newaxis, ...])
            inputs.append(split_input)
            labels.append(split_label)

        return np.concatenate(inputs, axis=0), np.concatenate(labels, axis=0)
    

def streamflow_dataset():
    df = pd.read_excel(
        io="./useful_volume.xlsx",
        header=0,
        index_col=0,
        parse_dates=True,
    )
    return df#.iloc[:, 0:3]

def get_uv_data():
    df = streamflow_dataset()
    scaler = MinMaxScaler()
    train_test_split_date = pd.Timestamp('2020-01-01')

    S_data = []

    for series_name in df.columns:
        series = df[[series_name]]
        df_norm = pd.DataFrame(scaler.fit_transform(series), index=series.index, columns=[series_name])
        window = WindowGenerator(1, 1, 1, [series_name])
        X, Y = window.make_dataset(df_norm)
        
        # Find rows with NaNs in X and Y to ensure continuity
        nan_rows_X = np.any(np.isnan(X), axis=1)
        nan_rows_Y = np.any(np.isnan(Y), axis=1)
        nan_rows = nan_rows_X | nan_rows_Y
        
        # Filter out the rows with NaNs
        X_clean, Y_clean = X[~nan_rows], Y[~nan_rows]

        S_data.append((X_clean, Y_clean))
      
    return S_data

def main():
    S_train = get_uv_data()
    X_train, Y_train = S_train[0]
    # X_test, Y_test = S_test[0]
    plt.plot(Y_train)
    plt.show()

if __name__ == "__main__":
    main()
