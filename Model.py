import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import gc
import sys
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM, RepeatVector, Concatenate, TimeDistributed, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lib.Utility import Utility
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

class ModelData:
    data_set:pd.DataFrame
    interval:int #minute

    def __init__(self, src_dir:str=None, detector_id:str=None, month_of_data:str=None, interval:int=1, with_rainfall:bool=True, specific_file:str=None):
        self.src_dir = src_dir
        self.detector_id = detector_id
        self.month_of_data = month_of_data
        self.interval = interval
        self.with_rainfall = with_rainfall
        self.specific_file = specific_file
        self.data_set = None
        self.load_data()

    def load_data(self, fill_limit=10):
        if self.data_set is not None:
            self.data_set.clear()
        file_list = []

        if self.specific_file is not None:
            if os.path.exists(self.specific_file):
                file_list = [self.specific_file]
            else:
                print("Warning: Specific file '{}' not found. Proceeding with all files.".format(self.specific_file))
        else:
            target_dir = "{}/{}".format(self.src_dir, self.detector_id)
            if self.month_of_data is not None:
                target_dir = "{}/{}".format(target_dir, self.month_of_data)
            
            file_list = Utility.list_file_by_pattern(target_dir, "*.csv")

        for file in file_list:
            try:
                df = pd.read_csv(file)
                df = df.sort_values('Date').reset_index(drop=True)
                if len(df) < 2:
                    print("Skipping file {}: Not enough data".format(file))
                    continue

                if self.data_set is None:
                    self.data_set = df
                else:
                    self.data_set = pd.concat([self.data_set, df], ignore_index=True)
            except Exception as e:
                print("Error loading file {}: {}".format(file, e))


        self.data_set['Date'] = pd.to_datetime(self.data_set['Date'].astype(str).str[:12], format='%Y%m%d%H%M')
        self.data_set.set_index('Date', inplace=True)
        # convert to 1min interval data
        self.data_set = self.data_set.groupby(self.data_set.index).mean()
        # resample to 1min interval, but only fill data within 10min
        self.data_set = self.data_set.resample('{}min'.format(self.interval)).mean().ffill(limit=fill_limit)
        # drop any remaining NaN rows (e.g., at start/end of file range)
        self.data_set.dropna(inplace=True)

        # Add time-based features
        self.data_set = ModelData.extract_features(self.data_set)
        self.data_set = ModelData.drop_unnecessary_columns(self.data_set, self.with_rainfall)

        self.scaled_data_set = self.data_set.values
        self.scaled_data_set = self.scaled_data_set.astype('float32')

        feature_list = self.get_features_list()
        for i in range(len(feature_list)):
            scaler = self.get_scaler(feature_list[i])
            target_col = self.scaled_data_set[:, i].reshape(-1, 1)
            scaled_col = scaler.transform(target_col)
            self.scaled_data_set[:, i] = scaled_col.flatten()

        pass

    @staticmethod
    def drop_unnecessary_columns(data_set:pd.DataFrame, with_rainfall:bool):
        drop_list = ['hour', 'dayofweek', 'IsWeekDay', 'IsPeakHour', 'IsOvernight', 'Month', 'IsHoliday']

        if with_rainfall == False:
            drop_list.append('Rainfall')
    
        for col in drop_list:
            if col not in data_set.columns:
                print("Warning: Column '{}' not found in data. Skipping drop.".format(col))
            else:
                data_set = data_set.drop(columns=[col])
        
        return data_set

    @staticmethod
    def extract_features(data_set:pd.DataFrame):
        data_set['hour'] = data_set.index.hour
        data_set['dayofweek'] = data_set.index.dayofweek
        data_set['Hour_sin'] = np.sin(2 * np.pi * data_set['hour'] / 24)
        data_set['Hour_cos'] = np.cos(2 * np.pi * data_set['hour'] / 24)
        data_set['DOW_sin'] = np.sin(2 * np.pi * data_set['dayofweek'] / 7)
        #data_set['DOW_cos'] = np.cos(2 * np.pi * data_set['dayofweek'] / 7)
        data_set['Speed_Diff'] = data_set['Speed'].diff().fillna(0)

        return data_set

    def get_features_list(self):
        return self.data_set.columns.tolist()

    def get_scaler(self, column):
        data_min = 0
        data_max = 100
        if column == "Occupancy":
            data_min = 0
            data_max = 1
        elif column in ["Hour_sin", "Hour_cos", "DOW_sin", "DOW_cos"]:
            data_min = -1
            data_max = 1 
        elif column in ["Speed_Diff"]:
            data_max = 50
            data_min = -50
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.data_min_ = np.array([data_min])
        scaler.data_max_ = np.array([data_max])
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
        scaler.n_samples_seen_ = 1
        scaler.min_ = -scaler.data_min_ * scaler.scale_
        return scaler

    def get_feature_list(self):
        all_indices = self.data_set.columns.tolist()

        id_map = dict()
        id_list = []
        for i in range(len(all_indices)):
            id_map[all_indices[i]] = i
            id_list.append(i)

        return id_list, id_map, all_indices

    def create_sequences(self, input_window, output_window, filter_date=None):
        features, feature_id_map, feature_names = self.get_feature_list()
        X, y = [], []
        length = len(self.data_set)
        
        if filter_date is not None:
            filter_date = pd.to_datetime(filter_date, format='%Y%m%d%H%M')

        for i in range(input_window, length - output_window + 1):
            
            # verify the data is continuous at 1‑minute intervals over the full window
            window_df = self.data_set.iloc[i - input_window:i + output_window]

            # the index should already be datetime; check that each step is exactly as self.interval
            diffs = window_df.index.to_series().diff().dropna()
            if not (diffs == pd.Timedelta("{}min".format(self.interval))).all():
                continue

            #check filter date, skip if exists
            if filter_date is not None and self.data_set.iloc[i].name != filter_date:
                continue
            
            window_df.index[0]
            past_seq = window_df.iloc[:input_window, features]
            target = window_df.iloc[input_window:input_window + output_window, feature_id_map["Speed"]]

            # convert to numpy immediately for consistency
            past_arr = past_seq.values
            target_arr = target.values
            
            # skip sequences containing NaNs
            if np.isnan(past_arr).any() or np.isnan(target_arr).any():
                continue

            X.append(past_arr)
            y.append(target_arr)

        return np.array(X), np.array(y)
    
    # convert an array of values into a dataset matrix
    def create_scaled_sequences(self, input_window:int, output_window:int):
        values = self.scaled_data_set.to_numpy() if isinstance(self.scaled_data_set, (pd.DataFrame, pd.Series)) else np.asarray(self.scaled_data_set)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        dataX, dataY, dataT = [], [], []
        target_col = self.get_features_list().index("Speed")
        max_start = len(values) - input_window - output_window + 1
        expected_step = pd.Timedelta(minutes=self.interval)

        for i in range(max_start):
            if output_window == 0:
                i = max_start - 1  # For web forecast, only create the last sequence

            seq_index = None
            if isinstance(self.data_set.index, pd.DatetimeIndex):
                seq_index = self.data_set.index[i:i + input_window + output_window]
                if len(seq_index) > 1:
                    step_diff = seq_index.to_series().diff().dropna()
                    if not (step_diff == expected_step).all():
                        continue
            x_window = values[i:i + input_window]
            y_window = values[i + input_window:i + input_window + output_window, target_col]
            if np.isnan(x_window).any() or np.isnan(y_window).any():
                continue
            dataX.append(x_window)
            dataY.append(y_window)
            if seq_index is not None:
                if output_window > 0:
                    dataT.append(seq_index[input_window:input_window + output_window].to_numpy())
                else:
                    #output for web forcast
                    dataT.append(seq_index[:input_window].to_numpy())
            else:
                dataT.append(np.array([None] * output_window))

            if output_window == 0:
                break  # Only create one sequence for web forecast

        X = np.array(dataX)
        y = np.array(dataY)
        timestamps = np.array(dataT)
        return X, y, timestamps

def cv_permutation_importance_lstm(X, y, feature_names, input_window, output_window, n_splits=5, n_repeats=5, random_state=42):
    """
    Cross-validated permutation importance for LSTM (3D input).
    - model_builder: a function that returns a fresh compiled LSTM model
    - Uses TimeSeriesSplit for proper temporal CV
    """
    np.random.seed(random_state)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    overall_importance = np.zeros(len(feature_names))
    overall_std = np.zeros(len(feature_names))
    
    print(f"Starting {n_splits}-fold TimeSeries CV for feature importance...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build and train a fresh model for this fold
        model = create_lstm_model(feature_names, input_window, output_window)   # IMPORTANT: fresh model each fold
        model, history = fit_model(model, X_train, y_train, X_val, y_val)

        # Baseline MAE on this validation fold
        y_pred_base = model.predict(X_val, verbose=0)
        baseline_mae = np.mean(np.abs(y_val.flatten() - y_pred_base.flatten()))
        
        fold_importance = []
        
        for feat_idx in range(X.shape[2]):
            scores = []
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                # Shuffle only this feature across samples (preserve time order)
                for t in range(X_val.shape[1]):
                    np.random.shuffle(X_perm[:, t, feat_idx])
                
                y_pred_perm = model.predict(X_perm, verbose=0)
                perm_mae = np.mean(np.abs(y_val.flatten() - y_pred_perm.flatten()))
                scores.append(perm_mae - baseline_mae)
            
            fold_importance.append(np.mean(scores))
        
        overall_importance += np.array(fold_importance)
        print(f"Fold {fold+1} completed.")
    
    # Average across folds
    overall_importance /= n_splits
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Mean': overall_importance
    }).sort_values(by='Importance Mean', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("CROSS-VALIDATED FEATURE IMPORTANCE (TimeSeriesSplit)")
    print("="*80)
    print(imp_df.round(5))
    
    return imp_df

def get_call_back():

    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, # stop after 10 rounds without improvement
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,   # learning rate will be reduced to 20% of the current value
        patience=5,    # 5 rounds without improvement will reduce the learning rate
        min_lr=0.0001
    )

    return [early_stop, reduce_lr]

def create_lstm_model(features, input_window, output_window):
    '''
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_window, len(features)), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_window))  # direct output_window future speeds
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae', 'mse'])

    '''
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_window, len(features)), return_sequences=True))
    # Second LSTM Layer (returns sequence)
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    # Third LSTM Layer (returns only final output)
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_window))
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mae',           # or tf.keras.losses.Huber(delta=1.0)
              metrics=['mae', 'mse'])
    
    return model

def fit_model(model, X_train, y_train):
    ''''
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,          # try 32–128
        callbacks=get_call_back(),
        shuffle=False           # CRITICAL for time series
    )
    '''

    history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2, callbacks=get_call_back())
    return model, history

def get_splited_data(model_data, input_window, output_window, train_size):
    # Create sequences from the scaled array rather than the dataframe
    X, y, y_timestamps = model_data.create_scaled_sequences(input_window, output_window)

    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    trainX, trainY = X[0:train_size,:], y[0:train_size,:]
    testX, testY = X[train_size:len(X),:], y[train_size:len(y),:]
    trainY_timestamps = y_timestamps[0:train_size,:]
    testY_timestamps = y_timestamps[train_size:len(y_timestamps),:]

    return trainX, trainY, trainY_timestamps, testX, testY, testY_timestamps

def train_model(detector_id, input_window, output_window, with_rainfall, start_year, end_year, interval, model_file_name):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = "{}/rsc/training_data/".format(working_dir)

    month_dir_list = os.listdir(src_dir + "/" + detector_id)
    trainY_timestamps = None
    testY_timestamps = None
    for month in month_dir_list:
        try:
            year = int(month[:4])
            if not (year >= start_year and year <= end_year):
                continue

            with_rainfall_str = "WithRainfall" if with_rainfall else "WithoutRainfall"
            print("Processing Detector:{} Month:{} Input Window:{} Output Window:{}".format(detector_id, month, input_window, output_window))
            model_file_path = "{}/model/traffic_speed_model_{}_{}_{}_{}_{}_{}_{}min.keras".format(working_dir, model_file_name, input_window, output_window, with_rainfall_str, start_year, end_year, interval)
            model_data = ModelData(src_dir=src_dir, detector_id=detector_id, month_of_data=month, interval=interval, with_rainfall=with_rainfall)
            trainX, trainY, trainY_timestamps, testX, testY, testY_timestamps = get_splited_data(model_data, input_window, output_window, train_size=0.8)

            model = None
            if os.path.exists(model_file_path):
                print("Loading existing model from {}".format(model_file_path))
                model = keras.models.load_model(model_file_path)
            else:
                features, feature_id_map, feature_names = model_data.get_feature_list()
                model = create_lstm_model(features, input_window, output_window)

            model, history = fit_model(model, trainX, trainY)

            # Save the model
            model.save(model_file_path)
            print("Model trained and saved as {}".format(model_file_path))

            # Predict
            # make predictions``
            trainPredict = model.predict(trainX)
            trainPredict = trainPredict.reshape(trainPredict.shape[0], output_window)
            testPredict = model.predict(testX)
            testPredict = testPredict.reshape(testPredict.shape[0], output_window)
            # invert predictions
            SpeedScaler = model_data.get_scaler("Speed")
            trainPredict_unscaled = SpeedScaler.inverse_transform(trainPredict)
            trainY_unscaled = SpeedScaler.inverse_transform(trainY)
            testPredict_unscaled = SpeedScaler.inverse_transform(testPredict)
            testY_unscaled = SpeedScaler.inverse_transform(testY)

            # calculate root mean squared error
            print('Train Score RMSE: %.2f' % (np.sqrt(mean_squared_error(trainY, trainPredict))))
            print('Train Score MAE: %.2f' % (mean_absolute_error(trainY, trainPredict)))
            print('Train Score Accuracy: %.2f' % ((1 - mean_absolute_percentage_error(trainY, trainPredict)) * 100))
            print()
            print('TEST Score MSE: %.4f' % (mean_squared_error(testY, testPredict)))
            print('Test Score RMSE: %.4f' % (np.sqrt(mean_squared_error(testY, testPredict))))
            print('Test Score MAE: %.4f' % (mean_absolute_error(testY, testPredict)))
            print('Test Score Accuracy: %.4f' % ((1 - mean_absolute_percentage_error(testY, testPredict)) * 100))

            

            
            # Plotting predicted vs actual speeds over time
            
            tmp_predict = list()
            tmp_answer = list()
            tmp_time = list()
            for i in range(0, len(testY_unscaled), output_window):
                tmp_predict.extend(testPredict_unscaled[i])
                tmp_answer.extend(testY_unscaled[i])
                tmp_time.extend(testY_timestamps[i])

            '''
            plot_font_size = 16
            plot_title_size = 18
            plt.plot(tmp_time, tmp_predict, color='orange', label='Prediction', alpha=0.8)
            plt.plot(tmp_time, tmp_answer, color='royalblue', label='Actual')
            plt.xlabel('Datetime (HKT)', fontsize=plot_font_size)
            plt.ylabel('Speed (km/h)', fontsize=plot_font_size)
            plt.title('Predicted vs Actual Speed over Time for Detector: {}  Model: Stacked LSTM\n\n'.format(detector_id), fontsize=plot_title_size)
            plt.legend(fontsize=plot_font_size)
            plt.gca().tick_params(axis='both', labelsize=plot_font_size)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gcf().autofmt_xdate()
            plt.grid(True, alpha=0.3)
            plt.show()
            '''
            
            pass

        except Exception as e:
            print("Error processing month {}: {}".format(month, repr(e)))

        gc.collect()  # Clean up memory after each month

    pass

if __name__ == "__main__":

    detector_id = sys.argv[1]
    input_window = int(sys.argv[2])
    output_window = int(sys.argv[3])
    with_rainfall = sys.argv[4].lower() == "true"
    start_year = int(sys.argv[5])
    end_year = int(sys.argv[6])
    interval = int(sys.argv[7])
    model_file_name = sys.argv[8]

    train_model(
        detector_id=detector_id,
        input_window=input_window,
        output_window=output_window,
        with_rainfall=with_rainfall,
        start_year=start_year,
        end_year=end_year,
        interval=interval,
        model_file_name=model_file_name
    )