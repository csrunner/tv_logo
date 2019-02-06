# -*- coding:utf-8 -*-
__author__ = 'shichao'


import os
import shutil

def del_two_end_files(in_root_dir,out_root_dir):

    for dir_h1 in os.listdir(in_root_dir):
        out_path_h1 = os.path.join(out_root_dir,dir_h1)
        in_path_h1 = os.path.join(in_root_dir,dir_h1)
        if not os.path.exists(out_path_h1):
            os.mkdir(out_path_h1)

        for dir_h2 in os.listdir(in_path_h1):
            out_path_h2 = os.path.join(out_path_h1,dir_h2)
            in_path_h2 = os.path.join(in_path_h1,dir_h2)
            if not os.path.exists(out_path_h2):
                os.mkdir(out_path_h2)

            for dir_h3 in os.listdir(in_path_h2): # tv_name
                out_path_h3 = os.path.join(out_path_h2, dir_h3)
                in_path_h3 = os.path.join(in_path_h2, dir_h3)
                if not os.path.exists(out_path_h3):
                    os.mkdir(out_path_h3)

                for dir_h4 in os.listdir(in_path_h3): # folders
                    out_path_h4 = os.path.join(out_path_h3, dir_h4)
                    in_path_h4 = os.path.join(in_path_h3, dir_h4)
                    if not os.path.exists(out_path_h4):
                        os.mkdir(out_path_h4)

                    for i in range(30,len([name for name in os.listdir(in_path_h4) if os.path.isfile(os.path.join(in_path_h4,name))])-30):
                        shutil.copy(os.path.join(in_path_h4,str(i).zfill(5)+'.jpg'),os.path.join(out_path_h4,str(i-30).zfill(5)+'.jpg'))


def main():
    input_dir = '/home/data/TVLOGO_IMG'
    output_dir = '/home/data/TVLOGO_CLEAN_IMG'
    del_two_end_files(input_dir,output_dir)

if __name__ == '__main__':
    main()