import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

class Lane:
    lane_id:str
    speed:int #km/hr
    occupancy:int
    volumn:int
    valid:bool

    def __init__(self, lane_id:str, speed:int, occupancy:int, volumn:str, valid:bool):
        self.lane_id = lane_id
        self.speed = speed
        self.occupancy = occupancy
        self.volumn = volumn
        self.valid = valid

class TrafficData:
    period_from:datetime
    period_to:datetime
    lanes:list

    def __init__(self, period_from:datetime, period_to:datetime):
        self.period_from = period_from
        self.period_to = period_to
        self.lanes = list()

    def add_lane(self, lane_id:str, speed:int, occupancy:int, volumn:str, valid:bool):
        self.lanes.append(Lane(lane_id, speed, occupancy, volumn, valid))

    def get_speed(self):
        sum = 0
        for l in self.lanes:
            sum += l.speed

        return sum / len(self.lanes)
    
    def get_occupancy(self):
        sum = 0
        for l in self.lanes:
            sum += l.occupancy

        return sum / len(self.lanes)
    
    def get_volumn(self):
        sum = 0
        for l in self.lanes:
            sum += l.volumn

        return sum / len(self.lanes)

class Detector:       
    detector_id:str
    direction:str
    lanes:dict
    latitude:float
    longitude:float
    name:str

    traffic_data:dict

    def __init__(self, name:str, detector_id:str, direction:str, latitude:float, longitude:float):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.detector_id = detector_id
        self.direction = direction
        self.traffic_data = dict()
    
    def add_traffic_data(self, traffic_data:TrafficData):
        self.traffic_data[traffic_data.period_from.strftime("%Y%m%d%H%M%S")] = traffic_data


class TrafficDataHandler:
    detector_info:dict
    detector_data:dict

    def __init__(self, detector_info_file:str):
        self.detector_data = dict()

        df = pd.read_csv(detector_info_file, skiprows=1, header=None, names=["detector_id", "district", "road_en" , "road_tc", "road_sc", "easting", "northing", "latitude", "longitude", "direction", "rotation"])
        
        self.detector_info = {
            detector_id: group.drop(columns='detector_id').to_dict(orient='list') 
            for detector_id, group in df.groupby('detector_id')
        }

        pass

    def load(self, traffic_data_file:str, filter_detector_id=None):
        try:
            with open(traffic_data_file) as f:
                
                xml_content = f.read()
                root = ET.fromstring(xml_content)
                date_str = root.findtext("date")
                
                for period_node in root.findall(".//period"):
                    from_str = period_node.findtext("period_from")
                    to_str   = period_node.findtext("period_to")

                    period_from = datetime.strptime(date_str + from_str, "%Y-%m-%d%H:%M:%S")
                    period_to = datetime.strptime(date_str + to_str, "%Y-%m-%d%H:%M:%S")

                    detector_nodes = None
                    if filter_detector_id is not None:
                        detector_nodes = period_node.findall(".//detector[detector_id='{}']".format(filter_detector_id))
                    else:
                        detector_nodes = period_node.findall("detectors/detector")

                    for detector_node in detector_nodes:
                        detector_id = detector_node.findtext("detector_id")

                        if filter_detector_id is not None and detector_id != filter_detector_id:
                            continue

                        if detector_id in self.detector_info:
                            if detector_id not in self.detector_data:
                                direction = detector_node.findtext("direction")
                                latitude = self.detector_info[detector_id]["latitude"][0]
                                longitude = self.detector_info[detector_id]["longitude"][0]
                                detector_name = self.detector_info[detector_id]["road_en"][0]
                                self.detector_data[detector_id] = Detector(detector_name, detector_id, direction, latitude, longitude)

                            traffic_data = TrafficData(period_from, period_to)
                            for lane_node in detector_node.findall("lanes/lane"):
                                lane_id = lane_node.findtext("lane_id")
                                speed = int(lane_node.findtext("speed"))
                                occupancy = int(lane_node.findtext("occupancy"))
                                volume = int(lane_node.findtext("volume"))
                                valid = False if lane_node.findtext("valid") == "N" else True

                                traffic_data.add_lane(lane_id, speed, occupancy, volume, valid)

                            self.detector_data[detector_id].add_traffic_data(traffic_data)

        except Exception as e:
            print("Read xml error. File:{} Msg:{}".format(traffic_data_file, repr(e)))


if __name__ == "__main__":
    d = TrafficDataHandler("D:/INTEGRATED_STUDY/rsc/traffic_speed_volume_occ_info.csv")
    d.load_traffic_data("D:/INTEGRATED_STUDY/rsc/traffic_data/20260201/20260201-0001-rawSpeedVol-all.xml")
    

    pass