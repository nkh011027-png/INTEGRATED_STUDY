# This file using to plot the relationship between traffic speed and rainfall, 
# and also plot the occupancy heatmap and time series boxplot for each detector.
# Used past traffic data and rainfall data from 2023 to 2025.

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import seaborn as sns
from scipy.spatial import KDTree
from scipy import interpolate
from matplotlib.colors import LogNorm
from lib.RainfallDataHandler import RainfallDataHandler
from lib.TrafficDataHandler import TrafficDataHandler
from lib.Utility import Utility
from datetime import datetime, timedelta
from matplotlib.image import NonUniformImage
from Model import ModelData
from keras import models

class TrafficVsRainfall():
    working_dir:str
    rainfall_src_dir:str
    traffic_src_dir:str
    training_src_dir:str
    start_time:datetime
    end_time:datetime
    title:str

    traffic_data_handler:TrafficDataHandler
    rainfall_data_handler:RainfallDataHandler

    def __init__(self, start_time_str, end_time_str, title):
        self.title = title
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.rainfall_src_dir = "{}/rsc/rainfall_nowcast_data/".format(self.working_dir)
        self.traffic_src_dir = "{}/rsc/traffic_data/".format(self.working_dir)
        self.training_src_dir = "{}/rsc/training_data/".format(self.working_dir)

        self.start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        
        self.traffic_data_handler = TrafficDataHandler("rsc/traffic_speed_volume_occ_info_filtered.csv")
        self.rainfall_data_handler = RainfallDataHandler()

        for i in range((self.end_time - self.start_time).days + 1):
            new_date = self.start_time + timedelta(days=i)

            rainfall_files = Utility.list_file_by_pattern("{}/{}".format(self.rainfall_src_dir, new_date.strftime("%Y%m")), "{}*.csv".format(new_date.strftime("%Y%m%d")))
            for f in rainfall_files:
                file_date = Utility.get_data_file_date(f)
                if self.start_time <= file_date <= self.end_time:
                    self.rainfall_data_handler.load(f)


            traffic_file = Utility.list_file_by_pattern("{}/{}".format(self.traffic_src_dir, new_date.strftime("%Y%m")), "{}*.xml".format(new_date.strftime("%Y%m%d")))
            for f in traffic_file:
                file_date = Utility.get_data_file_date(f)
                if self.start_time <= file_date <= self.end_time:
                    self.traffic_data_handler.load(f)
            pass
    
    def plot(self, detector_id):
        if detector_id not in self.traffic_data_handler.detector_data:
            return

        detector_lat = self.traffic_data_handler.detector_info[detector_id]['latitude'][0]
        detector_lon = self.traffic_data_handler.detector_info[detector_id]['longitude'][0]
        detector_road_name = self.traffic_data_handler.detector_info[detector_id]['road_en'][0]
        rainfall_iter = next(iter(self.rainfall_data_handler.rainfall_data_list.values()))
        distance, coord_index = Utility.get_nearest_lat_lon(rainfall_iter.latitude, rainfall_iter.longitude, detector_lat, detector_lon)
        
        time_data = list()
        rainfall_data = list()
        speed_data = list()
        occupancy_data = list()
        for t in self.rainfall_data_handler.rainfall_data_list:
            if t in self.traffic_data_handler.detector_data[detector_id].traffic_data:
                time_data.append(datetime.strptime(t, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M"))

                rainfall = 0
                for i in range(len(coord_index)):
                    rainfall += self.rainfall_data_handler.rainfall_data_list[t].rainfall[coord_index[i]]
                rainfall = rainfall / len(coord_index)

                rainfall_data.append(rainfall)
                speed_data.append(self.traffic_data_handler.detector_data[detector_id].traffic_data[t].get_speed())
                occupancy_data.append(self.traffic_data_handler.detector_data[detector_id].traffic_data[t].get_occupancy())

        plot_data = {
            'timestamp': pd.to_datetime(time_data),
            'traffic_speed': speed_data,  # Speed in km/h
            'rainfall_rate': rainfall_data, # Rainfall in mm/h
            'occupancy': occupancy_data # in percentage
        }

        plot_data = pd.DataFrame(plot_data)
        
        # Interpolate for smooth curves
        time_numeric = (plot_data['timestamp'] - plot_data['timestamp'].min()).dt.total_seconds()
        f_speed = interpolate.interp1d(time_numeric, plot_data['traffic_speed'], kind='cubic')
        f_occ = interpolate.interp1d(time_numeric, plot_data['occupancy'], kind='cubic')
        time_interp = np.linspace(time_numeric.min(), time_numeric.max(), num=1000)
        speed_interp = f_speed(time_interp)
        occ_interp = f_occ(time_interp)
        time_interp_dt = plot_data['timestamp'].min() + pd.to_timedelta(time_interp, unit='s')
        
        # 2. Setup the Figure and Axes
        fig, ax1 = plt.subplots(figsize=(20, 8))
        # Adjust the right margin to make room for the third axis
        fig.subplots_adjust(right=0.8)
        # Scale text size by 1.5 (assuming default 12, so 18)
        default_fontsize = 17

        # 3. Plot Traffic Speed (Primary Y-Axis - Left)
        color_speed = 'tab:blue'
        ax1.set_xlabel('Hong Kong Time (UTC+8)', fontsize=default_fontsize)
        ax1.set_ylabel('Traffic speed (km/h)', color=color_speed, fontsize=default_fontsize)
        ax1.plot(time_interp_dt, speed_interp, color=color_speed, label='Speed')
        ax1.tick_params(axis='y', labelcolor=color_speed, labelsize=default_fontsize)
        ax1.tick_params(axis='x', labelsize=default_fontsize)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # 4. Plot Occupancy (Secondary Y-Axis - Right)
        ax2 = ax1.twinx()
        color_occ = 'tab:red'
        ax2.set_ylabel('Road occupancy (%)', color=color_occ, fontsize=default_fontsize)
        ax2.plot(time_interp_dt, occ_interp, color=color_occ, linestyle='--', label='Occupancy')
        ax2.tick_params(axis='y', labelcolor=color_occ, labelsize=default_fontsize)
        ax2.set_ylim(0, 100)

        # 5. Plot Rainfall (Third Y-Axis - Offset Right)
        ax3 = ax1.twinx()
        # Offset the right spine to prevent overlap with Occupancy axis
        ax3.spines['right'].set_position(('outward', 60)) 
        color_rain = 'tab:gray'
        ax3.set_ylabel('Half-hourly rainfall (mm)', color=color_rain, fontsize=default_fontsize)
        # Using a bar plot for rainfall is common for better visibility
        ax3.bar(plot_data['timestamp'], plot_data['rainfall_rate'], color=color_rain, alpha=0.3, width=0.01, label='Rainfall')
        ax3.tick_params(axis='y', labelcolor=color_rain, labelsize=default_fontsize)
        ax3.set_ylim(0, 80)

        # 6. Final Formatting
        plt.title('Traffic Dynamics vs. Rainfall Rate ({})\n\n{}({})\n\nFrom {}HKT to {}HKT'.format(
            self.title, 
            detector_road_name,
            detector_id,
            self.start_time.strftime("%Y-%m-%d %H:%M"), 
            self.end_time.strftime("%Y-%m-%d %H:%M"),
            ), fontsize=default_fontsize, pad=20)
        
        fig.tight_layout()
        plt.savefig("plot/TrafficVsRainfall_{}_{}_to_{}.png".format(detector_id, self.start_time.strftime("%Y%m%d%H%M"), self.end_time.strftime("%Y%m%d%H%M")))
        fig.clear()
        plt.close()
        gc.collect()
        #plt.show()

class TrafficVsRainfallVsPredicted():
    working_dir:str
    rainfall_src_dir:str
    traffic_src_dir:str
    training_src_dir:str
    start_time:datetime
    end_time:datetime
    title:str

    traffic_data_handler:TrafficDataHandler
    rainfall_data_handler:RainfallDataHandler

    def __init__(self, title, detector_id, month_of_data, date_str):
        self.title = title
        self.working_dir = os.path.dirname(os.path.abspath(__file__))

        self.training_src_dir = "{}/rsc/training_data/".format(self.working_dir)
        self.model = models.load_model("{}/{}/traffic_speed_model_general_12_18_WithRainfall_2025_2025_5min.keras".format(self.working_dir, "model"))
        
        
        model_data = ModelData(self.training_src_dir, detector_id, month_of_data, interval=5)
        X, y = model_data.create_sequences(input_window=12, output_window=18, filter_date=date_str)
        
        data_time = model_data.data_set.iloc[218 - 12:218 + 18]

        self.Predicted_y = self.model.predict(X, verbose=0)

        self.rainfall_src_dir = "{}/rsc/rainfall_nowcast_data/".format(self.working_dir)
        self.traffic_src_dir = "{}/rsc/traffic_data/".format(self.working_dir)
        self.training_src_dir = "{}/rsc/training_data/".format(self.working_dir)

        #self.start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        #self.end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        
        self.traffic_data_handler = TrafficDataHandler("rsc/traffic_speed_volume_occ_info_filtered.csv")
        self.rainfall_data_handler = RainfallDataHandler()

        for i in range((self.end_time - self.start_time).days + 1):
            new_date = self.start_time + timedelta(days=i)

            rainfall_files = Utility.list_file_by_pattern("{}/{}".format(self.rainfall_src_dir, new_date.strftime("%Y%m")), "{}*.csv".format(new_date.strftime("%Y%m%d")))
            for f in rainfall_files:
                file_date = Utility.get_data_file_date(f)
                if self.start_time <= file_date <= self.end_time:
                    self.rainfall_data_handler.load(f)


            traffic_file = Utility.list_file_by_pattern("{}/{}".format(self.traffic_src_dir, new_date.strftime("%Y%m")), "{}*.xml".format(new_date.strftime("%Y%m%d")))
            for f in traffic_file:
                file_date = Utility.get_data_file_date(f)
                if self.start_time <= file_date <= self.end_time:
                    self.traffic_data_handler.load(f)
            pass
    
    def plot(self, detector_id):
        if detector_id not in self.traffic_data_handler.detector_data:
            return

        detector_lat = self.traffic_data_handler.detector_info[detector_id]['latitude'][0]
        detector_lon = self.traffic_data_handler.detector_info[detector_id]['longitude'][0]
        detector_road_name = self.traffic_data_handler.detector_info[detector_id]['road_en'][0]
        rainfall_iter = next(iter(self.rainfall_data_handler.rainfall_data_list.values()))
        distance, coord_index = Utility.get_nearest_lat_lon(rainfall_iter.latitude, rainfall_iter.longitude, detector_lat, detector_lon)
        
        time_data = list()
        rainfall_data = list()
        speed_data = list()
        occupancy_data = list()
        for t in self.rainfall_data_handler.rainfall_data_list:
            if t in self.traffic_data_handler.detector_data[detector_id].traffic_data:
                time_data.append(datetime.strptime(t, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M"))

                rainfall = 0
                for i in range(len(coord_index)):
                    rainfall += self.rainfall_data_handler.rainfall_data_list[t].rainfall[coord_index[i]]
                rainfall = rainfall / len(coord_index)

                rainfall_data.append(rainfall)
                speed_data.append(self.traffic_data_handler.detector_data[detector_id].traffic_data[t].get_speed())
                occupancy_data.append(self.traffic_data_handler.detector_data[detector_id].traffic_data[t].get_occupancy())

        plot_data = {
            'timestamp': pd.to_datetime(time_data),
            'traffic_speed': speed_data,  # Speed in km/h
            'rainfall_rate': rainfall_data, # Rainfall in mm/h
            'occupancy': occupancy_data # in percentage
        }

        plot_data = pd.DataFrame(plot_data)
        
        # Interpolate for smooth curves
        time_numeric = (plot_data['timestamp'] - plot_data['timestamp'].min()).dt.total_seconds()
        f_speed = interpolate.interp1d(time_numeric, plot_data['traffic_speed'], kind='cubic')
        f_occ = interpolate.interp1d(time_numeric, plot_data['occupancy'], kind='cubic')
        time_interp = np.linspace(time_numeric.min(), time_numeric.max(), num=1000)
        speed_interp = f_speed(time_interp)
        occ_interp = f_occ(time_interp)
        time_interp_dt = plot_data['timestamp'].min() + pd.to_timedelta(time_interp, unit='s')
        
        # 2. Setup the Figure and Axes
        fig, ax1 = plt.subplots(figsize=(20, 8))
        # Adjust the right margin to make room for the third axis
        fig.subplots_adjust(right=0.8)
        # Scale text size by 1.5 (assuming default 12, so 18)
        default_fontsize = 17

        # 3. Plot Traffic Speed (Primary Y-Axis - Left)
        color_speed = 'tab:blue'
        ax1.set_xlabel('Hong Kong Time (UTC+8)', fontsize=default_fontsize)
        ax1.set_ylabel('Traffic speed (km/h)', color=color_speed, fontsize=default_fontsize)
        ax1.plot(time_interp_dt, speed_interp, color=color_speed, label='Speed')
        ax1.tick_params(axis='y', labelcolor=color_speed, labelsize=default_fontsize)
        ax1.tick_params(axis='x', labelsize=default_fontsize)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # 4. Plot Occupancy (Secondary Y-Axis - Right)
        ax2 = ax1.twinx()
        color_occ = 'tab:red'
        ax2.set_ylabel('Road occupancy (%)', color=color_occ, fontsize=default_fontsize)
        ax2.plot(time_interp_dt, occ_interp, color=color_occ, linestyle='--', label='Occupancy')
        ax2.tick_params(axis='y', labelcolor=color_occ, labelsize=default_fontsize)
        ax2.set_ylim(0, 100)

        # 5. Plot Rainfall (Third Y-Axis - Offset Right)
        ax3 = ax1.twinx()
        # Offset the right spine to prevent overlap with Occupancy axis
        ax3.spines['right'].set_position(('outward', 60)) 
        color_rain = 'tab:gray'
        ax3.set_ylabel('Half-hourly rainfall (mm)', color=color_rain, fontsize=default_fontsize)
        # Using a bar plot for rainfall is common for better visibility
        ax3.bar(plot_data['timestamp'], plot_data['rainfall_rate'], color=color_rain, alpha=0.3, width=0.01, label='Rainfall')
        ax3.tick_params(axis='y', labelcolor=color_rain, labelsize=default_fontsize)
        ax3.set_ylim(0, 80)

        # 6. Final Formatting
        plt.title('Traffic Dynamics vs. Rainfall Rate ({})\n\n{}({})\n\nFrom {}HKT to {}HKT'.format(
            self.title, 
            detector_road_name,
            detector_id,
            self.start_time.strftime("%Y-%m-%d %H:%M"), 
            self.end_time.strftime("%Y-%m-%d %H:%M"),
            ), fontsize=default_fontsize, pad=20)
        
        fig.tight_layout()
        plt.savefig("plot/TrafficVsRainfall_{}_{}_to_{}.png".format(detector_id, self.start_time.strftime("%Y%m%d%H%M"), self.end_time.strftime("%Y%m%d%H%M")))
        fig.clear()
        plt.close()
        gc.collect()
        #plt.show()

class CSVPlotter():
    working_dir:str
    training_src_dir:str
    data_set:pd.DataFrame

    def __init__(self):
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.training_src_dir = "{}/rsc/training_data/".format(self.working_dir)
        self.data_set = None

    def load_csv(self, file:str):
        data = pd.read_csv(file)
        data['Date'] = pd.to_datetime(data['Date'].astype(str), format='%Y%m%d%H%M%S')
        if self.data_set is None:
            self.data_set = data
        else:
            self.data_set = pd.concat([self.data_set, data], ignore_index=True)
            pass

    def plot_occupancy_heatmap(self, detector_id, threadshold=25, start_year=None, end_year=None):
        
        min_month = None
        max_month = None
        month_dir_list = os.listdir(self.training_src_dir + "/" + detector_id)
        for month in month_dir_list:
            if start_year is not None and month[:4] < start_year:
                continue
            if end_year is not None and month[:4] > end_year:
                continue
            if min_month is None or month < min_month:
                min_month = month
            if max_month is None or month > max_month:
                max_month = month
            file_list = Utility.list_file_by_pattern("{}/{}/{}".format(self.training_src_dir, detector_id, month), "{}*.csv".format(month))
            for f in file_list:
                self.load_csv(f)

        if self.data_set is None or self.data_set.empty:
            print("No data loaded for heatmap.")
            return
        self.data_set = self.data_set[self.data_set['Occupancy'] >= threadshold]
        # Extract time of day and occupancy
        self.data_set['hour_minute'] = self.data_set['Date'].dt.hour + self.data_set['Date'].dt.minute / 60.0  # 0-24
        occupancy_data = self.data_set['Occupancy']
        hour_minute = self.data_set['hour_minute']
        max_occ = round(occupancy_data.max())
        min_occ = round(occupancy_data.min())

        # Bin occupancy into 20 bins from 0 to max_occ
        occupancy_bins = np.linspace(min_occ, max_occ, max_occ - min_occ + 1)
        #occupancy_binned = pd.cut(occupancy_data, bins=occupancy_bins, labels=False)

        # Create 2D histogram
        time_bins = np.linspace(0, 24, 48)  # Half-hourly bins (48 bins)
        hist, xedges, yedges = np.histogram2d(hour_minute, occupancy_data, bins=[time_bins, occupancy_bins])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(20, 8))
        im = ax.imshow(hist.T, interpolation='nearest', origin='lower', aspect='auto', cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=hist.max()), extent=[time_bins[0], time_bins[-1], occupancy_bins[0], occupancy_bins[-1]])
        ax.set_xlabel('Time of Day (UFC+8)', fontsize=18)
        ax.set_ylabel('Occupancy (%)', fontsize=18)
        ax.set_title(f'Occupancy Heatmap for {detector_id} ({min_month[:4]}/{min_month[4:6]} to {max_month[:4]}/{max_month[4:6]})', fontsize=21, pad=20)
        #ax.set_xticks(time_bins)  # Convert to bin indices
        #ax.set_yticks(occupancy_bins)
        #ax.set_xticklabels([f"{i:02d}" for i in range(24)], fontsize=17)
        #ax.set_xticklabels([f"{i:02d}" for i in range(24)], fontsize=17)
        #ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
        time_bins = np.linspace(0, 24, 24)
        ax.set_xticks(time_bins)  # Convert to bin indices
        ax.set_xticklabels([f"{int(i):02d}" for i in time_bins], fontsize=17)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=17)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency', fontsize=18)

        plt.tight_layout()
        #plt.show()
        plt.savefig(f"plot/OccupancyHeatmap_{detector_id}_{min_month}_to_{max_month}.png")
        fig.clear()
        plt.close()
        gc.collect()

    def plot_occupancy_time_series_boxplot_hourly(self, detector_id, start_year=None, end_year=None):
        
        min_month = None
        max_month = None
        month_dir_list = os.listdir(self.training_src_dir + "/" + detector_id)
        for month in month_dir_list:
            if start_year is not None and month[:4] < start_year:
                continue
            if end_year is not None and month[:4] > end_year:
                continue
            if min_month is None or month < min_month:
                min_month = month
            if max_month is None or month > max_month:
                max_month = month
            file_list = Utility.list_file_by_pattern("{}/{}/{}".format(self.training_src_dir, detector_id, month), "{}*.csv".format(month))
            for f in file_list:
                self.load_csv(f)

        if self.data_set is None or self.data_set.empty:
            print("No data loaded for time series boxplot.")
            return

        # Extract time of day and occupancy
        self.data_set['hour_minute'] = self.data_set['Date'].dt.hour + self.data_set['Date'].dt.minute / 60.0  # 0-24
        occupancy_data = self.data_set['Occupancy'].round(0)
        hour_minute = self.data_set['hour_minute']

        # Bin time into hour intervals
        hour_bins = np.arange(0, 25, 1)  # 0, 1, 2, ..., 24
        self.data_set['hour_bin'] = pd.cut(hour_minute, bins=hour_bins, labels=hour_bins[:-1], right=False)

        # Group by hour bin
        grouped = self.data_set.groupby('hour_bin')['Occupancy']

        # Prepare data for boxplot: list of arrays for each bin
        box_data = [group.values for name, group in grouped if not group.empty]

        # Positions for boxplot (midpoints of bins)
        positions = hour_bins[:-1] + 0.5  # e.g., 0.5, 1.5, ..., 23.5

        fig, ax = plt.subplots(figsize=(20, 8))
        bp = ax.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)  # showfliers=False to hide outliers
        
        # Customize: color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        for whisker in bp['whiskers']:
            whisker.set_color('black')
        for cap in bp['caps']:
            cap.set_color('black')

        '''
        # Add markers for min and max
        for i, group in enumerate(box_data):
            if len(group) > 0:
                max_val = np.max(group)
                ax.scatter(positions[i], max_val, color='red', marker='o', s=50, zorder=3, label='Max Occupancy' if i == 0 else "")
        '''

        ax.set_xlabel('Time of Day (UTC+8)', fontsize=17)
        ax.set_ylabel('Occupancy (%)', fontsize=17)
        ax.set_title(f'Occupancy Time Series Boxplot by Hour for {detector_id} ({min_month[:4]}/{min_month[4:6]} to {max_month[:4]}/{max_month[4:6]})', fontsize=21, pad=20)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{i:02d}" for i in range(24)], fontsize=17)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=17)
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)

        # Add legend for markers
        ax.legend(loc='upper right')

        plt.tight_layout()
        #plt.show()
        plt.savefig(f"plot/OccupancyTimeSeriesBoxplotHourly_{detector_id}_{min_month}_to_{max_month}.png")
        fig.clear()
        plt.close()
        gc.collect()

    
        
    
    
if __name__ == "__main__":
    #CSVPlotter().plot_occupancy_heatmap("AID04212", threadshold=25, start_year="2023", end_year="2025")
    #CSVPlotter().plot_occupancy_time_series_boxplot_hourly("AID04212", start_year="2023", end_year="2025")
    #p = TrafficVsRainfall("2023-09-07 00:00:00", "2023-09-08 23:59:59", "century black rainstorm")
    #p = TrafficVsRainfall("2023-09-01 00:00:00", "2023-09-02 23:59:59", "super typhoon Saola")
    #p = TrafficVsRainfall("2023-10-08 00:00:00", "2023-10-09 23:59:59", "typhoon Koinu")
    #p = TrafficVsRainfall("2025-07-29 00:00:00", "2025-07-29 14:59:59", "black rainstorm")
    #p = TrafficVsRainfall("2025-08-02 00:00:00", "2025-08-02 23:59:59", "black rainstorm")
    #p = TrafficVsRainfall("2025-08-14 00:00:00", "2025-08-14 23:59:59", "typhoon Podul")
    #p.plot("AID07119")
    #p.plot("AID04212")
    #p.plot("AID04120")
    #p.plot("AID01110")
    #p.plot("AID07106")
    #p.plot("AID07104")
    #p.plot("AID03108")
    #p.plot("AID03209")

    #p = TrafficVsRainfall("2023-09-07 00:00:00", "2023-09-08 23:59:59", "century black rainstorm")
    #p.plot("AID04212")
    #p = TrafficVsRainfall("2023-09-01 00:00:00", "2023-09-02 23:59:59", "super typhoon Saola")
    #p.plot("AID04212")
    #p = TrafficVsRainfall("2023-10-08 09:00:00", "2023-10-09 21:59:59", "typhoon Koinu")
    #p.plot("AID04212")
    #p = TrafficVsRainfall("2025-07-29 00:00:00", "2025-07-29 14:59:59", "black rainstorm")
    #p.plot("AID04212")
    #p = TrafficVsRainfall("2026-03-02 00:00:00", "2026-03-02 23:59:59", "")
    #p.plot("AID04212")

    p = TrafficVsRainfallVsPredicted("Traffic Speed vs Rainfall vs Predicted Speed", "AID04212", "202309", "202309011800")

    pass