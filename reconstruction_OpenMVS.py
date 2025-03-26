# data_folder_Structure
# ├── mask
# │   ├── 0000000000.png
# │   ├── 0000000001.png
# │   ├── ...
# ├── other
# │   ├── 0000000000.png
# │   ├── 0000000001.png
# │   ├── ...
# ├── output

from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import calculate_zoom as calz
import noiseReduction as nr

def reconstrucionFromImages(pathstring):
    # pathstring = "/media/xujx/MyPassport/sfm-pro/20221028-1-T154-0/"

    pathstring += "/"
    mask_dir_string = os.path.join(pathstring, 'mask')
    other_dir_string = os.path.join(pathstring, 'other')
    path_outputs_string = os.path.join(pathstring, 'output/')

    # openMVS
    converter = "cd /usr/local/bin/OpenMVS"
    converter = converter + "\n"

    converter = converter + "./InterfaceCOLMAP -i "
    converter = converter + path_outputs_string
    converter = converter + " -o "
    converter = converter + path_outputs_string + "scene_panicle.mvs"
    converter = converter + " --image-folder "
    converter = converter + mask_dir_string
    converter = converter + "\n"

    converter = converter + "./DensifyPointCloud -w "
    converter = converter + path_outputs_string
    converter = converter + " -i scene_panicle.mvs -o panicle.mvs --remove-dmaps 1"
    converter = converter + "\n"

    converter = converter + "./InterfaceCOLMAP -i "
    converter = converter + path_outputs_string
    converter = converter + " -o "
    converter = converter + path_outputs_string + "scene_other.mvs"
    converter = converter + " --image-folder "
    converter = converter + other_dir_string
    converter = converter + "\n"

    converter = converter + "./DensifyPointCloud -w "
    converter = converter + path_outputs_string
    converter = converter + " -i scene_other.mvs -o other.mvs --remove-dmaps 1"
    converter = converter + "\n"

    os.system(converter)

    return 0

def process_folders(root_path, folder):
    folder_path = os.path.join(root_path, folder)
    video_folder_path = folder_path
    images_output_path = os.path.join(folder_path, 'images')

    if os.path.exists(video_folder_path) and os.path.isdir(video_folder_path):
        # 创建images文件夹，如果不存在
        if not os.path.exists(images_output_path):
            os.makedirs(images_output_path)
        for video_name in os.listdir(video_folder_path):
            video_file_path = os.path.join(video_folder_path, video_name)
            video_to_frames(video_file_path, images_output_path)

def check_dense_scene_ply(folder_path):
    ply_file_path = os.path.join(folder_path, 'output', 'dense_scene.ply')
    return os.path.exists(ply_file_path)

def traverse_folders(root_path):
    folders_to_exclude = []
    # check
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path) and check_dense_scene_ply(folder_path):
            print(f"Excluding folder name: {folder_name}")
            folders_to_exclude.append(folder_name)

    filtered_folders = [folder_name for folder_name in os.listdir(root_path) if folder_name not in folders_to_exclude]
    return filtered_folders

if __name__ == '__main__':

    path = '/your/path'
    reconstrucionFromImages(path)

