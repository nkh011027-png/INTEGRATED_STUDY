import glob
import os
import numpy as np
from scipy.spatial import KDTree
from datetime import datetime

class Utility:

    @staticmethod
    def list_file_by_pattern(path:str, pattern:str, sort_desc=False):
        search_pattern = os.path.join(path, pattern)
        files = glob.glob(search_pattern)
        files.sort(reverse=sort_desc)
        return files
    
    @staticmethod
    def get_data_file_date(file:str):
        split_str = os.path.basename(file).split("-")
        return datetime.strptime("{}{}".format(split_str[0], split_str[1]), "%Y%m%d%H%M")
    
    @staticmethod
    def get_nearest_lat_lon(lat_array, lon_array, lat, lon, k=1):
        lat_array_copy = lat_array[:]
        lon_array_copy = lon_array[:]

        distance_list = list()
        index_list = list()

        for i in range(k):
            points = np.column_stack((lat_array_copy, lon_array_copy))
            tree = KDTree(points)
            distance, index = tree.query([lat, lon])

            distance_list.append(distance)
            index_list.append(index)

            lat_array_copy.pop(index)
            lon_array_copy.pop(index)

        return distance_list, index_list
    
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        absolute_diff = np.abs(array - value)
        idx = absolute_diff.argmin()
        return array[idx]
    