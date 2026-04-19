# This program used to unzip the downloaded zip files, and move the files to the corresponding folders.

import glob
import zipfile
import os
import shutil
from os.path import basename

path = r"D:\Downloads\*.zip"
zip_files = glob.glob(path)

for file in zip_files:
    with zipfile.ZipFile(file, 'r') as zip_ref:

        dirname = basename(file).split("-")[1].split(".")[0]

        output_dir = r"D:\INTEGRATED_STUDY\rainfall_nowcast_data\\" + dirname
        if os.path.exists(output_dir) == False:
            os.mkdir(output_dir)

        contents = zip_ref.namelist()
        contents.pop(0)
        for item in contents:
            item_dir = item.split("/")[0]
            #print(item)
            if os.path.exists(output_dir + "\\" + basename(item)) == False:
                zip_ref.extract(item, path=output_dir)
                shutil.move(output_dir + "\\" + item, output_dir)
                shutil.rmtree(output_dir + "\\" + item_dir)
