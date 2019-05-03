#-*- coding:utf-8 -*-
__author__ = 'shichao'

import os
import shutil
#/home/shichao/data/data_bak/TVLOGO_IMG_ALL
source_root_path = './data/data_bak/TVLOGO_IMG_ALL'
dest_root_path = './data/TV_LOGO_ALL_MERGE'
station_type_list = os.listdir(source_root_path)

for station_type in station_type_list:
    station_type_path = os.path.join(source_root_path,station_type)
    img_type_list = os.listdir(station_type_path)

    for img_type in img_type_list:
        img_type_path = os.path.join(station_type_path,img_type)
        station_name_list = os.listdir(img_type_path)

        for station_name in station_name_list:
            counter = 0
            dst_station_path = os.path.join(dest_root_path,station_name)

            station_name_path = os.path.join(img_type_path,station_name)
            sub_folder_list = os.listdir(station_name_path)

            for sub_folder in sub_folder_list:
                sub_folder_path = os.path.join(station_name_path,sub_folder)
                file_list = os.listdir(sub_folder_path)

                for file_name in file_list:
                    if os.path.isdir(os.path.join(sub_folder_path,file_name)):
                        child_folder_path = os.path.join(sub_folder_path,file_name)
                        child_file_list = os.listdir(child_folder_path)

                        for child_file_name in child_file_list:
                            child_file_path = os.path.join(child_folder_path,child_file_name)
                            ext = os.path.splitext(child_file_name)[1]
                            file_new_path = os.path.join(dst_station_path,str(counter).zfill(5)+ext)
                        try:

                            shutil.copyfile(file_path,file_new_path)
                            counter += 1

                        except IOError:
                            if not os.path.exists(dst_station_path):
                                os.mkdir(dst_station_path)
                            shutil.copyfile(file_path,file_new_path)
                            counter += 1

                    else:
                        ext = os.path.splitext(file_name)[1]

                        file_path = os.path.join(sub_folder_path,file_name)
                        # print(file_path)
                        print(os.path.join(sub_folder_path,file_name))
                        print(os.path.join(dst_station_path,str(counter).zfill(5)+ext))
                        file_new_path = os.path.join(dst_station_path,str(counter).zfill(5)+ext)
                        # file_new_path = os.rename(os.path.join(sub_folder_path,file_name),os.path.join(dst_station_path,str(counter).zfill(5)+ext))
                        try:
                        
                            shutil.copyfile(file_path,file_new_path)
                            counter += 1
                                                
                        except IOError:
                            if not os.path.exists(dst_station_path):
                                os.mkdir(dst_station_path)
                            shutil.copyfile(file_path,file_new_path)
                            counter += 1


