import os
import sys
import json
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tensorflow as tf
import schedule
import time
import logging
import gc
import holidays
from datetime import datetime
from pathlib import Path
from lib.Utility import Utility
from lib.RainfallDataHandler import RainfallDataHandler
from lib.TrafficDataHandler import TrafficDataHandler, TrafficData
from Model import ModelData

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CODE = "general"  

# Configuration
TRAFFIC_URL = "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"
RAINFALL_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"
TRAFFIC_SAVE_DIR = f"{SCRIPT_DIR}/website/past_data/traffic"
RAINFALL_SAVE_DIR = f"{SCRIPT_DIR}/website/past_data/rainfall"
COMBINED_SAVE_DIR = f"{SCRIPT_DIR}/website/past_data/combined"
FORECAST_SAVE_DIR = f"{SCRIPT_DIR}/website/forcast_data"
PREDICTED_SPEED_OUTPUT_FILE = f"{FORECAST_SAVE_DIR}/predicted_traffic_speed.json"
DETECTOR_INFO_FILE = f"{SCRIPT_DIR}/rsc/traffic_speed_volume_occ_info_filtered.csv"
MODEL_DIR = f"{SCRIPT_DIR}/model"

# Store coord_index cache to avoid recalculating
coord_index_cache = {}
hk_holidays = holidays.HongKong()

class RainfallDataCustomLoader:
    # Custom rainfall data loader for the actual CSV format from HKO

    rainfall_data_list:dict

    def __init__(self):
        self.rainfall_data_list = {}

    def load(self, file):
        # Load rainfall data from CSV file with actual HKO format 
        try:
            df = pd.read_csv(file, skiprows=1, header=None, 
                            names=['update_time', 'end_time', 'latitude', 'longitude', 'rainfall'])

            if df.empty:
                logger.warning(f"No data in rainfall file: {file}")
                return False

            # Group by end_time to organize the data
            for end_time_val, group in df.groupby('end_time'):
                try:
                    # Convert end_time to datetime
                    end_time_str = str(int(end_time_val)) + "00"
                    end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M%S")
        
                    # Get update time from first row of this group
                    update_time_str = str(int(group['update_time'].iloc[0])) + "00"
                    update_time = datetime.strptime(update_time_str, "%Y%m%d%H%M%S")
        
                    # Extract latitude, longitude, and rainfall as lists
                    latitude = group['latitude'].tolist()
                    longitude = group['longitude'].tolist()
                    rainfall = group['rainfall'].tolist()
        
                    # Store in the format expected by the rest of the code
                    key = end_time.strftime("%Y%m%d%H%M%S")
        
                    from lib.RainfallDataHandler import RainfallData
                    self.rainfall_data_list[key] = RainfallData(
                        update_time=update_time,
                        end_time=end_time,
                        latitude=latitude,
                        longitude=longitude,
                        rainfall=rainfall
                    )
                    logger.debug(f"Loaded rainfall data for {key}: {len(latitude)} points")
    
                except Exception as e:
                    logger.debug(f"Error processing rainfall group {end_time_val}: {e}")
                    continue

            logger.info(f"Successfully loaded {len(self.rainfall_data_list)} rainfall time periods from {file}")
            return True

        except Exception as e:
            logger.error(f"Error loading rainfall file {file}: {e}")
            return False

class DataDownloader:
    # Handle downloading and processing traffic and rainfall data 
    
    def __init__(self):
        # Initialize the downloader 
        self.model_cache = {}
        self.model_index_cache = None
        self._create_directories()
        self.detector_info = self._load_detector_info()

    def _ensure_directory(self, directory):
        # Ensure a path is a directory; if a file exists, move it away first 
        if os.path.exists(directory) and not os.path.isdir(directory):
            backup_path = f"{directory}.bak_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.replace(directory, backup_path)
            logger.warning(f"Moved file to {backup_path} because {directory} must be a directory")
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _create_directories(self):
        # Create necessary directories if they don't exist 
        for directory in [TRAFFIC_SAVE_DIR, RAINFALL_SAVE_DIR, COMBINED_SAVE_DIR, FORECAST_SAVE_DIR]:
            self._ensure_directory(directory)

    def _parse_model_metadata(self, model_path):
        # Parse model settings encoded in filename 
        filename = os.path.basename(model_path)
        parts = filename.split("_")
        if len(parts) < 10:
            return None

        try:
            model_code = parts[3]
            input_window = int(parts[4])
            output_window = int(parts[5])
            with_rainfall = parts[6] == "WithRainfall"
            interval = int(parts[9].replace("min.keras", ""))
            return {
                "model_code": model_code,
                "input_window": input_window,
                "output_window": output_window,
                "with_rainfall": with_rainfall,
                "interval": interval,
                "model_path": model_path,
            }
        except Exception:
            return None

    def _build_model_index(self):
        # Build model code -> model metadata index, choosing latest file per model code
        model_dir = MODEL_DIR
        if not os.path.isdir(model_dir):
            return {}

        index = {}
        for file in os.listdir(model_dir):
            #if not file.endswith(".keras"):
            if not file.endswith("traffic_speed_model_general_12_18_WithRainfall_2025_2025_5min.keras"):
                continue

            model_path = os.path.join(model_dir, file)
            metadata = self._parse_model_metadata(model_path)
            if metadata is None:
                continue

            model_code = metadata["model_code"]
            if model_code not in index:
                index[model_code] = metadata

        return index

    def _load_model_cached(self, model_code, model_path):
        # Load keras model once and reuse in future cycles (cache key: model_code) 
        if model_code not in self.model_cache:
            self.model_cache[model_code] = tf.keras.models.load_model(model_path)
        return self.model_cache[model_code]

    def preload_all_models(self, force_reload=False):
        # Preload all detector model files into memory cache 
        if self.model_index_cache is None:
            self.model_index_cache = self._build_model_index()

        if not self.model_index_cache:
            logger.warning("No model files found for preloading")
            return {"total": 0, "loaded": 0, "skipped": 0, "failed": 0}

        loaded = 0
        skipped = 0
        failed = 0

        for model_code, model_meta in self.model_index_cache.items():
            model_path = model_meta["model_path"]
            logger.info(f"Preloading model for {model_code}")
            if not force_reload and model_code in self.model_cache:
                skipped += 1
                continue

            try:
                self.model_cache[model_code] = tf.keras.models.load_model(model_path)
                loaded += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to preload model for {model_code} from {model_path}: {e}")

        total = len(self.model_index_cache)
        logger.info(
            f"Model preload completed - total: {total}, loaded: {loaded}, skipped: {skipped}, failed: {failed}"
        )
        return {"total": total, "loaded": loaded, "skipped": skipped, "failed": failed}

    def _prepare_detector_dataframe(self, csv_file, interval, fill_limit=10, with_rainfall=True, input_window=None):
        # Load and preprocess detector historical data to align with training format 
        df = pd.read_csv(csv_file)
        if df.empty or "Date" not in df.columns:
            return None
        
        # Parse full timestamp and align it to interval buckets.
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        if df.empty:
            return None

        # Round to the nearest interval-aligned timestamp (for example nearest 5-min boundary).
        df["Date"] = df["Date"].dt.round(f"{interval}min")

        df = df.sort_values("Date").reset_index(drop=True)
        df.set_index("Date", inplace=True)
        df = df.groupby(df.index).mean()
        df = df.resample(f"{interval}min").mean().ffill(limit=fill_limit)
        if not df.empty:
            aligned_mask = (
                (df.index.minute % interval == 0)
                & (df.index.second == 0)
                & (df.index.microsecond == 0)
            )
            df = df.loc[aligned_mask]
        df.dropna(inplace=True)

        if input_window is not None and input_window > 0 and not df.empty:
            latest_time = pd.Timestamp.now().round(f"{interval}min")
            latest_time = pd.Timestamp("2026-04-13 06:50:00")
            cutoff_time = latest_time - pd.Timedelta(minutes=(interval * max(input_window - 1, 0)) + 15)
            df = df[df.index >= cutoff_time]

        


        if df.empty or len(df.index) < input_window:
            return None

        df = ModelData.extract_features(df)
        df = ModelData.drop_unnecessary_columns(df, with_rainfall=with_rainfall)  # Keep rainfall column for prediction input, even if model doesn't use it, to maintain feature alignment

        return df

    def _build_prediction_inputs(self, df, input_window, output_window, with_rainfall, interval):
        # Build one inference sample using the latest available detector history 
        all_features = df.columns.tolist()
        ''''
        all_features = [
            "IsWeekDay",
            "IsHoliday",
            "IsPeakHour",
            "IsOvernight",
            "Month",
            "Occupancy",
            "Volumn",
            "Speed",
            "Rainfall",
            "Hour",
            "Minute",
        ]
        '''


        past_features = all_features[:]
        if not with_rainfall and "Rainfall" in past_features:
            past_features.remove("Rainfall")

        #future_features = [x for x in past_features if x not in ["Occupancy", "Volumn", "Speed"]]

        if len(df) < input_window:
            return None, None, None, None, None

        past_df = df.tail(input_window)
        past_times = list(past_df.index)
        past_speeds = past_df["Speed"].astype(float).tolist() if "Speed" in past_df.columns else []
        last_time = df.index[-1]
        future_times = [last_time + pd.Timedelta(minutes=interval * step) for step in range(1, output_window + 1)]

        '''
        last_rainfall = 0.0
        if "Rainfall" in df.columns and len(df) > 0 and pd.notna(df["Rainfall"].iloc[-1]):
            last_rainfall = float(df["Rainfall"].iloc[-1])

        
        future_rows = []
        for ts in future_times:
            future_rows.append(
                {
                    "IsWeekDay": 1 if ts.weekday() < 5 else 0,
                    "IsHoliday": 1 if ts.date() in hk_holidays else 0,
                    "Rainfall": last_rainfall
                }
            }
        '''

        #future_df = pd.DataFrame(future_rows, index=future_times)
        x_past = np.expand_dims(past_df[past_features].astype(np.float32).values, axis=0)
        #x_future = np.expand_dims(future_df[future_features].astype(np.float32).values, axis=0)
        #return x_past, x_future, future_times, past_times, past_speeds
        return x_past, future_times, past_times,past_speeds

    def predict_traffic_speed(self):
        # Predict future traffic speed for all detectors and save a single JSON file 
        if self.model_index_cache is None:
            self.model_index_cache = self._build_model_index()

        # get all combined CSV files
        combined_files = Utility.list_file_by_pattern(COMBINED_SAVE_DIR, "*.csv")

        all_predictions = {}
        default_output_window = 18
        default_interval = 5
         

        # append predictions while keeping timestamps aligned
        def append_prediction(detector_id, ts, speed):
            if detector_id not in all_predictions:
                all_predictions[detector_id] = {"timestamp": [], "predicted_speed": []}
            all_predictions[detector_id]["timestamp"].append(pd.to_datetime(ts).strftime("%Y%m%d%H%M%S"))
            all_predictions[detector_id]["predicted_speed"].append(round(float(speed), 1))

        #load csv files
        detector_data = {}
        for csv_file in combined_files:
            try:
                detector_id = os.path.basename(csv_file).replace(".csv", "")
                model_code = MODEL_CODE if MODEL_CODE is not None else detector_id
                model_meta = self.model_index_cache.get(model_code)

                model_data = ModelData(specific_file=csv_file, interval=model_meta["interval"], with_rainfall=model_meta["with_rainfall"])
                X, y, times = model_data.create_scaled_sequences(input_window=model_meta["input_window"], output_window=0)
                detector_data[detector_id] = (X, times[0], model_data.get_scaler("Speed"), model_meta)
                logger.info(f"Prepared model input for {detector_id} with {len(times[0])} time steps")
            except Exception as e:
                logger.warning(f"Failed to create model data for {csv_file}: {e}")
                continue

        #find the latest timestamp across all detectors to align predictions
        latest_timestamp = None
        for detector_id, data in detector_data.items():
            X, times, scaler, model_meta = data
            if times.size > 0:
                max_time = times.max()
                if latest_timestamp is None or max_time > latest_timestamp:
                    latest_timestamp = max_time

        #generate predictions for each detector, aligning them to the latest timestamp found
        all_predictions = {}
        for detector_id, data in detector_data.items():
            try:
                X, past_time, scaler, model_meta = data
                if past_time.size > 0 and past_time.max() == latest_timestamp:
                    if len(X[0]) == model_meta["input_window"] and len(past_time) == model_meta["input_window"]:
                        model = self._load_model_cached(model_meta["model_code"], model_meta["model_path"])
                        y_pred = model.predict(X, verbose=0)
                        predicted_speeds = scaler.inverse_transform(y_pred).reshape(-1).astype(float).tolist()
                        past_speeds = scaler.inverse_transform(X[0][:, model_data.get_features_list().index("Speed")].reshape(1, model_meta["input_window"]))[0]
                        last_time = past_time[-1]
                        forecast_times = [
                            last_time + pd.Timedelta(minutes=model_meta["interval"] * step) for step in range(1, model_meta["output_window"] + 1)
                        ]

                        #for ts, speed in zip(past_time, past_speeds):
                        #    append_prediction(detector_id, ts, speed)

                        for ts, speed in zip(forecast_times, predicted_speeds):
                            append_prediction(detector_id, ts, speed)

                else:
                    logger.warning(f"Skipping prediction for {detector_id} due to no valid timestamps or misaligned latest timestamp {latest_timestamp}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate predictions for {detector_id}: {e}")
                continue


        # Keep per-detector arrays aligned and ordered by time.
        for detector_id in all_predictions:
            pairs = list(
                zip(
                    all_predictions[detector_id]["timestamp"],
                    all_predictions[detector_id]["predicted_speed"],
                )
            )
            pairs.sort(key=lambda x: x[0])
            all_predictions[detector_id]["timestamp"] = [p[0] for p in pairs]
            all_predictions[detector_id]["predicted_speed"] = [p[1] for p in pairs]

        with open(PREDICTED_SPEED_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)

        total_records = sum(len(v["timestamp"]) for v in all_predictions.values())
        logger.info(f"Saved {total_records} predicted records to {PREDICTED_SPEED_OUTPUT_FILE}")
    
    def _load_detector_info(self):
        # Load detector information from CSV file# 
        try:
            df = pd.read_csv(DETECTOR_INFO_FILE)
            detector_info = {}
            for _, row in df.iterrows():
                detector_id = row['AID_ID_Number']
                detector_info[detector_id] = {
                    'latitude': float(row['Latitude']),
                    'longitude': float(row['Longitude']),
                    'road_en': row['Road_EN']
                }
            from typing import Dict, Tuple, List
            logger.info(f"Loaded {len(detector_info)} detectors from {DETECTOR_INFO_FILE}")
            return detector_info
        except Exception as e:
            logger.error(f"Error loading detector info: {e}")
            return {}
    
    def download_traffic_data(self, timestamp_str):
        # Download traffic data and save with timestamp# 
        try:
            logger.info(f"Downloading traffic data...")
            response = requests.get(TRAFFIC_URL, timeout=30)
            response.raise_for_status()
            
            filename = f"traffic_data_{timestamp_str}.xml"
            filepath = os.path.join(TRAFFIC_SAVE_DIR, filename)
            
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Traffic data saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error downloading traffic data: {e}")
            return None
    
    def download_rainfall_data(self, timestamp_str):
        # Download rainfall data and save with timestamp# 
        try:
            logger.info(f"Downloading rainfall data...")
            response = requests.get(RAINFALL_URL, timeout=30)
            response.raise_for_status()
            
            filename = f"rainfall_data_{timestamp_str}.csv"

            filename = f"{timestamp_str[:8]}-{timestamp_str[8:16]}-rainfall_data.csv"
            filepath = os.path.join(RAINFALL_SAVE_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            #with open(filepath, 'w', encoding='utf-8') as f:
            # Use custom loader for rainfall data
            #rainfall_loader = RainfallDataCustomLoader()
            #success = rainfall_loader.load(filepath)

            #if not success:
            #    logger.warning("Failed to load rainfall data with custom loader")
            #    return False

            #rainfall_handler.rainfall_data_list = rainfall_loader.rainfall_data_list
            
            logger.info(f"Rainfall data saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error downloading rainfall data: {e}")
            return None
    
    def combine_data(self, traffic_file, rainfall_file, timestamp_str):
        # Combine traffic and rainfall data and save to detector-specific CSV files# 
        try:
            # Load data
            traffic_handler = TrafficDataHandler(DETECTOR_INFO_FILE)
            rainfall_handler = RainfallDataHandler()
            
            logger.info(f"Loading traffic data from {traffic_file}")
            traffic_handler.load(traffic_file, filter_detector_id=None)
            
            logger.info(f"Loading rainfall data from {rainfall_file}")
            rainfall_handler.load(rainfall_file)
            
            if len(traffic_handler.detector_data) == 0:
                logger.warning("No traffic data loaded")
                return False
            
            if len(rainfall_handler.rainfall_data_list) == 0:
                logger.warning("No rainfall data loaded")
                return False
            
            rainfall_date_list = list(map(int, list(rainfall_handler.rainfall_data_list.keys())))
            
            # Process each detector
            for detector_id in traffic_handler.detector_data:
                try:
                    detector_lat = self.detector_info[detector_id]['latitude']
                    detector_lon = self.detector_info[detector_id]['longitude']
                    
                    # Calculate coord_index if not cached
                    if detector_id not in coord_index_cache:
                        rainfall_iter = next(iter(rainfall_handler.rainfall_data_list.values()))
                        _, coord_index = Utility.get_nearest_lat_lon(
                            rainfall_iter.latitude,
                            rainfall_iter.longitude,
                            detector_lat,
                            detector_lon
                        )
                        coord_index_cache[detector_id] = coord_index[0]
                    
                    coord_idx = coord_index_cache[detector_id]
                    
                    # Get or create CSV file
                    csv_file = os.path.join(COMBINED_SAVE_DIR, f"{detector_id}.csv")
                    
                    # Initialize CSV with header if it doesn't exist
                    if not os.path.exists(csv_file):
                        with open(csv_file, 'w') as f:
                            f.write("Date,IsWeekDay,IsHoliday,IsPeakHour,IsOvernight,Month,Occupancy,Speed,Volumn,Rainfall")
                    
                    # Read existing records to check if we need to append
                    existing_df = pd.read_csv(csv_file)

                    # Build new records to append
                    new_records = []
                    for t_date in traffic_handler.detector_data[detector_id].traffic_data:
                        try:
                            r_date = Utility.find_nearest(rainfall_date_list, int(t_date))
                            t_datetime = datetime.strptime(t_date, "%Y%m%d%H%M%S")
                            t_datetime_key = t_datetime.strftime("%Y%m%d%H%M%S")

                            if int(t_datetime_key) not in existing_df['Date'].values:
                                r_data = rainfall_handler.rainfall_data_list[str(r_date)]
                                r_start_date = r_data.update_time
                                r_end_date = r_data.end_time
                                
                                if (r_start_date <= t_datetime <= r_end_date and 
                                    coord_idx < len(r_data.rainfall)):
                                    
                                    traffic_data = traffic_handler.detector_data[detector_id].traffic_data[t_date]
                                    
                                    write_var = []
                                    write_var.append(t_datetime_key)
                                    write_var.append(1 if t_datetime.weekday() < 5 else 0)  # IsWeekDay
                                    write_var.append(0)  # IsHoliday - would need holidays library
                                    write_var.append(1 if t_datetime.hour in [7, 8, 9, 17, 18, 19] else 0)  # IsPeakHour
                                    write_var.append(1 if t_datetime.hour in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5] else 0)  # IsOvernight
                                    write_var.append(t_datetime.month)
                                    write_var.append(round(traffic_data.get_occupancy(), 1))
                                    write_var.append(round(traffic_data.get_speed(), 1))
                                    write_var.append(round(traffic_data.get_volumn(), 1))
                                    write_var.append(r_data.rainfall[coord_idx])
                                    write_var = list(map(str, write_var))
                                    new_records.append(",".join(write_var))
                        except Exception as e:
                            logger.debug(f"Error processing record for {detector_id} at {t_date}: {e}")
                            continue
                    
                    # Append new records to CSV
                    if new_records:
                        with open(csv_file, 'a') as f:
                            for record in new_records:
                                f.write("\n" + record)
                        
                        # Keep only last 2880 records (1 day)
                        with open(csv_file, 'r') as f:
                            all_lines = f.readlines()
                        
                        if len(all_lines) > 2881:  # 1 header + 2880 data rows
                            header = all_lines[0]
                            data_lines = all_lines[-(2880):]  # Keep last 2880 records
                            
                            with open(csv_file, 'w') as f:
                                f.write(header)
                                f.writelines(data_lines)
                        
                        #logger.info(f"Updated {detector_id}.csv with {len(new_records)} new records")
                
                except Exception as e:
                    logger.error(f"Error processing detector {detector_id}: {e}")
                    continue
            
            return True
        
        except Exception as e:
            logger.error(f"Error combining data: {e}")
            return False
    
    def download_and_process(self):
        # Warm up model cache once to reduce per-detector model load latency.
        if len(self.model_cache) == 0:
            self.preload_all_models()

        
        # Main function to download and process data# 
        try:
            #house keep
            house_keep_path = [TRAFFIC_SAVE_DIR, RAINFALL_SAVE_DIR]
            for p in house_keep_path:
                file_list = Utility.list_file_by_pattern(p, "*")
                for f in file_list:
                    try:
                        os.remove(f)
                        logger.debug(f"Deleted old file: {f}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {f}: {e}")
                        continue

            # Get current timestamp
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d%H%M")
            
            logger.info(f"Starting download cycle at {timestamp_str}")
            
            # Download both files with the same timestamp
            traffic_file = self.download_traffic_data(timestamp_str)
            rainfall_file = self.download_rainfall_data(timestamp_str)
            
            # If both downloads succeed, combine the data
            if traffic_file and rainfall_file:
                logger.info("Both files downloaded successfully, combining data...")
                success = self.combine_data(traffic_file, rainfall_file, timestamp_str)
                if success:
                    logger.info(f"Data processing completed successfully at {timestamp_str}")
                else:
                    logger.warning(f"Data combination failed at {timestamp_str}")
            else:
                logger.warning("One or both downloads failed, skipping data combination")
        
        except Exception as e:
            logger.error(f"Error in download_and_process: {e}")
        

        # Predict traffic speed for each detector using the trained model
        try:
            self.predict_traffic_speed()
        except Exception as e:
            logger.error(f"Error in traffic speed prediction: {e}")

        gc.collect()


def schedule_downloads(interval_minutes=5):
    # Schedule downloads to run at regular intervals 
    downloader = DataDownloader()
    
    logger.info(f"Starting scheduled downloads every {interval_minutes} minute(s)")
    
    # Schedule the job
    #schedule.every(interval_minutes).minutes.at(":00").do(downloader.download_and_process)
    for minute in range(0, 60, interval_minutes):
        # Format as ":00", ":05", etc.
        schedule.every().hour.at(f":{minute:02d}").do(downloader.download_and_process)


    # Keep scheduler running
    try:
        while True:
            schedule.run_pending()
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


def run_once():
    # Run the download and processing once for testing 
    downloader = DataDownloader()
    downloader.download_and_process()


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            # Run once for testing
            logger.info("Running download and processing once...")
            run_once()
        elif sys.argv[1] == "--schedule":
            # Run on schedule
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            schedule_downloads(interval)
        else:
            print("Usage:")
            print("  python traffic_forcast.py --once              # Run once for testing")
            print("  python traffic_forcast.py --schedule [minutes] # Schedule downloads (default: 1 minute)")
    else:
        # Default: run on 1-minute schedule
        schedule_downloads(5)
        #run_once()
