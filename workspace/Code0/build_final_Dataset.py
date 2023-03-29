import os
import sys
import shutil
from random import shuffle
import cv2
import numpy as np
import yaml


def get_min_data_num(dir_path):        
    """
    dir_path/class1
    dir_path/class2
    dir_path/class3
    dir_path/class4
    brief: 获得四个类之中最少的文件数量
    """       
    son_path = os.listdir(dir_path)
    min_num = 1e06
    for son in son_path:
        num = len(os.listdir(os.path.join(dir_path, son)))
        if num < min_num:
            min_num = num

    return min_num


def getDatasetInfo(dir_path):
    """
    dir_path/class1
    dir_path/class2
    dir_path/class3
    dir_path/class4
    brief: 获得四个类之中最少的文件数量
    """       
    dataset_info = {}
    son_path = os.listdir(dir_path)
    for son in son_path:
        if son not in dataset_info:
            dataset_info[son] = []

        num = len(os.listdir(os.path.join(dir_path, son)))
        dataset_info[son].append(num)
    print(dataset_info)
    
    
def filt_img(path, limit=30):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)               # 读图
    if image is None:
        return False
    min_side = min(image.shape[:2])
    if min_side <= limit:
        return False
    else:
        return True


def split_dataset(dir_path, val_ratio, train_ratio):
    son_path = os.listdir(dir_path)
    # SON: head, person, noveh, veh
    for son in son_path:
        cur_root_path = os.path.join(dir_path, son)
        file_names = os.listdir(cur_root_path)
        file_num = len(file_names)
        
        if file_num > 40000:
            file_num = int(file_num * 0.7)
        #elif file_num > 20000:
        #    file_num = int(file_num * 0.2)
        #elif file_num > 15000:
        #    file_num = int(file_num * 0.266)
        #elif file_num > 10000:
        #    file_num = int(file_num * 0.4)
        #elif file_num > 5000:
        #    file_num = int(file_num * 1)
        
        #file_names = file_names[:file_num]
        shuffle(file_names)
        val_num = int(file_num * val_ratio - 1)
        train_num = int(file_num * train_ratio - 1)
        # all is file names ; isn't paths
        train_files = file_names[:train_num]
        # print(f"son: {}")
        for train_file in train_files:
            src_file_path = os.path.join(cur_root_path, train_file)
            if False == filt_img(src_file_path):
                continue
            file_name = os.path.basename(src_file_path)
            dst_root_path = os.path.join(train_path, son)
            if not os.path.exists(dst_root_path):
                os.mkdir(dst_root_path)
            
            dst_file_path = os.path.join(dst_root_path, file_name)
            print(f"src: {src_file_path}; \ndst: {dst_file_path}")
            shutil.copy2(src_file_path, dst_file_path)


        val_files = file_names[train_num:]
        for val_file in val_files:
            src_file_path = os.path.join(cur_root_path, val_file)
            if False == filt_img(src_file_path):
                continue
            file_name = os.path.basename(src_file_path)
            dst_root_path = os.path.join(val_path, son)
            if not os.path.exists(dst_root_path):
                os.mkdir(dst_root_path)
            dst_file_path = os.path.join(dst_root_path, file_name)
            print(f"src: {src_file_path}; \ndst: {dst_file_path}")
            shutil.copy2(src_file_path, dst_file_path)


if __name__ == "__main__":
    val_ratio = 0.1
    train_ratio = 0.9
    scale = 0.2

    cfg_file = "data_config/dataset_info.yaml"
    with open(cfg_file, encoding='UTF-8') as f:
        file_stream = yaml.load(f, yaml.FullLoader)

    dataset_name = file_stream["finally_dataset_path"]
    root_dir = file_stream["subDataset_save_root"]
    num_class = file_stream["class_num"]
    class_token = f"{num_class}class_"
    print(f"dataset_num: {dataset_name}, class_token: {class_token}, root_dir: {root_dir}")

    exit()
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    train_path = f"{dataset_name}/train"                     
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    val_path = f"{dataset_name}/val"
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    dataset_paths = [
        
    ]
    
    datasets = os.listdir(root_dir)
    for dataset in datasets:
        if dataset[:7] == class_token:
            dataset_paths.append(os.path.join(root_dir, dataset))
            

    for dataset_path in dataset_paths:
        print(f"dataset: {dataset_path}")
        # getDatasetInfo(dataset_path)

        split_dataset(dataset_path, val_ratio, train_ratio)