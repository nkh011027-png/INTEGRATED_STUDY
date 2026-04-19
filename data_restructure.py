import os
import holidays
from lib.Utility import Utility
from lib.RainfallDataHandler import RainfallDataHandler
from lib.TrafficDataHandler import TrafficDataHandler, TrafficData
from datetime import datetime, timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(days=n)

def write_csv(csv_file, pending_write):
    with open(csv_file, "w") as file:
        #print("Write {}".format(csv_file))
        file.write("Date,IsWeekDay,IsHoliday,IsPeakHour,IsOvernight,Month,Occupancy,Speed,Volumn,Rainfall\n")
        file.write("\n".join(pending_write))

if __name__ == "__main__":
    hk_holidays = holidays.HongKong()
    filter_detector_id = None
    start_date = datetime.strptime("20220701", "%Y%m%d")
    end_date = datetime.strptime("20260219", "%Y%m%d")

    for date in daterange(start_date, end_date):
        print("Process Date: {}".format(date))
        rainfall_src_dir = "rsc/rainfall_nowcast_data/{}".format(date.strftime("%Y%m"))
        traffic_src_dir = "rsc/traffic_data/{}".format(date.strftime("%Y%m"))

        traffic_data_handler = TrafficDataHandler("rsc/traffic_speed_volume_occ_info_filtered.csv")
        rainfall_data_handler = RainfallDataHandler()

        start = datetime.now()

        rainfall_files = Utility.list_file_by_pattern(rainfall_src_dir, "{}*.csv".format(date.strftime("%Y%m%d")))
        for f in rainfall_files:
            rainfall_data_handler.load(f)

        if len(rainfall_data_handler.rainfall_data_list) == 0:
            continue

        traffic_files = Utility.list_file_by_pattern(traffic_src_dir, "{}*.xml".format(date.strftime("%Y%m%d")))
        for f in traffic_files:
            traffic_data_handler.load(f, filter_detector_id)

        if len(traffic_data_handler.detector_data) == 0:
            continue

        rainfall_date_list = list(map(int, list(rainfall_data_handler.rainfall_data_list.keys())))

        for detector_id in traffic_data_handler.detector_data:
            if filter_detector_id is not None and detector_id != filter_detector_id:
                continue

            #print("Detector {}".format(detector_id))
            
            train_data_dir = "rsc/training_data/{}/{}".format(detector_id, date.strftime("%Y%m"))
            os.makedirs(train_data_dir, exist_ok=True)

            csv_file = "{}/{}.csv".format(train_data_dir, date.strftime("%Y%m%d"))
            if os.path.exists(csv_file):
                continue
            
            
            detector_lat = traffic_data_handler.detector_info[detector_id]['latitude'][0]
            detector_lon = traffic_data_handler.detector_info[detector_id]['longitude'][0]
            detector_road_name = traffic_data_handler.detector_info[detector_id]['road_en'][0]
            rainfall_iter = next(iter(rainfall_data_handler.rainfall_data_list.values()))
            distance, coord_index = Utility.get_nearest_lat_lon(rainfall_iter.latitude, rainfall_iter.longitude, detector_lat, detector_lon)
            
            pending_write = list()
            for t_date in traffic_data_handler.detector_data[detector_id].traffic_data:
                
                r_date = Utility.find_nearest(rainfall_date_list, int(t_date))

                t_datetime = datetime.strptime(t_date, "%Y%m%d%H%M%S")
                r_start_date = rainfall_data_handler.rainfall_data_list[str(r_date)].update_time
                r_end_date = rainfall_data_handler.rainfall_data_list[str(r_date)].end_time

                if r_start_date <= t_datetime <= r_end_date and coord_index[0] < len(rainfall_data_handler.rainfall_data_list[str(r_date)].rainfall):
                    traffic_data:TrafficData = traffic_data_handler.detector_data[detector_id].traffic_data[t_date]
                    
                    write_var = list()
                    write_var.append(t_datetime.strftime("%Y%m%d%H%M%S"))
                    write_var.append(1 if date.weekday() < 5 else 0)
                    write_var.append(1 if date in hk_holidays else 0)
                    write_var.append(1 if t_datetime.hour in [7,8,9,17,18,19] else 0)
                    write_var.append(1 if t_datetime.hour in [20,21,22,23,0,1,2,3,4,5] else 0)
                    write_var.append(date.month)
                    write_var.append(round(traffic_data.get_occupancy(),1))
                    write_var.append(round(traffic_data.get_speed(),1))
                    write_var.append(round(traffic_data.get_volumn(),1))
                    write_var.append(rainfall_data_handler.rainfall_data_list[str(r_date)].rainfall[coord_index[0]])
                    write_var = list(map(str, write_var))
                    pending_write.append(",".join(write_var))
                    pass
                else:
                    pass
                    #print("not found. t_date:{} r_start:{} r_end:{}".format(t_datetime, r_start_date,r_end_date))
            
            write_csv(csv_file, pending_write)

    pass