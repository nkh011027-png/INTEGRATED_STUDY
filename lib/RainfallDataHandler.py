import numpy as np
import pandas as pd
from datetime import datetime
from .Utility import Utility

class RainfallData:
    update_time:datetime
    end_time:datetime
    latitude:list
    longitude:list
    rainfall:list
    coordinate_rainfall_map:dict

    def __init__(self, update_time, end_time, latitude, longitude, rainfall):
        self.update_time = update_time
        self.end_time = end_time
        self.latitude = latitude
        self.longitude = longitude
        self.rainfall = rainfall
        self.coordinate_rainfall_map = dict(zip(zip(latitude, longitude), rainfall))


class RainfallDataHandler:
    rainfall_data_list:dict

    def __init__(self):
        self.rainfall_data_list = dict()
    
    def load(self, file, nearest=True):
        try:
            df = pd.read_csv(file, skiprows=1, header=None, names=['update_time', 'end_time', 'latitude', 'longitude', 'rainfall'])
            file_datetime = Utility.get_data_file_date(file)

            split_data = {
                time: group.drop(columns='end_time').to_dict(orient='list') 
                for time, group in df.groupby('end_time')
            }

            for t in split_data:
                update_time = datetime.strptime(str(split_data[t]["update_time"][0]) + "00", "%Y%m%d%H%M%S")
                end_time = datetime.strptime(str(int(t)) + "00", "%Y%m%d%H%M%S")

                if t not in self.rainfall_data_list or self.rainfall_data_list[t].update_time < update_time:
                    rainfall_data = RainfallData(
                        update_time = update_time,
                        end_time = end_time,
                        latitude = split_data[t]["latitude"],
                        longitude = split_data[t]["longitude"],
                        rainfall = split_data[t]["rainfall"],
                    )

                    self.rainfall_data_list[file_datetime.strftime("%Y%m%d%H%M%S")] = rainfall_data

                if nearest:
                    break
        except Exception as e:
            print("Read csv error. file:{} Msg:{}".format(file, repr(e)))

if __name__ == "__main__":
    handler = RainfallDataHandler()
    handler.load("D:/INTEGRATED_STUDY/rsc/rainfall_nowcast_data/20260206/20260206-0100-Gridded_rainfall_nowcast.csv")
    pass