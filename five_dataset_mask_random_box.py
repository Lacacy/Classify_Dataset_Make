import os
import sys
import shutil
import numpy as np
from random import shuffle
import cv2


"""
1. 主要做了从无标定数据的图片之中生成 Mask;
2. 分割 Mask 为 train和val;
"""

def getCenterPoint(size):
    """
    输入图像宽高, 返回随机中心点:
    """
    if size is not None:
        h = max(20, size[0])
        w = max(20, size[1])

        return (np.random.randint(0, h - 1), np.random.randint(0, w - 1))
    else:
        return (0, 0)


def generateBox(img_size, rand_range_wh, box_num):
    boxes = []
    if img_size is not None:
        for i in range(box_num):
            h = max(20, img_size[0])
            w = max(20, img_size[1])
            centerP = getCenterPoint((h, w))
            rand_w = np.random.randint(*rand_range_wh)
            rand_h = np.random.randint(*rand_range_wh)
            min_x = max(0, centerP[1] - rand_w // 2)
            min_y = max(0, centerP[0] - rand_h // 2)
            max_x = min(w - 1, centerP[1] + rand_w // 2)
            max_y = min(h - 1, centerP[0] + rand_h // 2)
            boxes.append([min_x, min_y, max_x, max_y])
        return boxes
    else:
        return boxes
    
    
def writeBox(img, box, save_path):
    if img is not None:
        save_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # print("save_path", save_path)
        cv2.imwrite(save_path, save_img)

        
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
    
    
def getExplicitPostfixFiles(path, file_paths,
    control_suffix_list=["jpg", "png", "bmp", "JPEG", "jpeg", "jfif", "tiff"]):
    '''
    : brief: 递归获取指定后缀的所有文件
    :param path: file_paths
    :return:
    '''
    for i in os.listdir(path):
        new = os.path.join(path, i)
        if os.path.isfile(new):
            if new[new.rfind(".") + 1:] in control_suffix_list:
                file_paths.append(new)
        elif os.path.isdir(new):
            getExplicitPostfixFiles(new, file_paths)


def split_dataset(dir_path, val_ratio, train_ratio):
    """
    brief: 仅仅用于生成Mask数据;
    1. 递归获取给定路径所有的文件
    2. 将所有文件移动到指定目录
    """
    file_names = []
    getExplicitPostfixFiles(dir_path, file_names)
    file_num = len(file_names)
    dir_name = "Mask"
    file_names = file_names[:file_num]
    shuffle(file_names)
    val_num = int(file_num * val_ratio - 1)
    train_num = int(file_num * train_ratio - 1)
    # all is file names ; isn't paths
    train_files = file_names[:train_num]
    # print(f"son: {}")
    for train_file in train_files:
        src_file_path = train_file
        file_name = os.path.basename(src_file_path)
        dst_root_path = os.path.join(train_path, dir_name)
        if not os.path.exists(dst_root_path):
                os.mkdir(dst_root_path)
        dst_file_path = os.path.join(dst_root_path, file_name)
        print(f"src: {src_file_path}; \ndst: {dst_file_path}")
        shutil.copy2(src_file_path, dst_file_path)

    val_files = file_names[train_num:]
    for val_file in val_files:
        src_file_path = val_file
        file_name = os.path.basename(src_file_path)
        dst_root_path = os.path.join(val_path, dir_name)
        if not os.path.exists(dst_root_path):
            os.mkdir(dst_root_path)
        dst_file_path = os.path.join(dst_root_path, file_name)
        print(f"src: {src_file_path}; \ndst: {dst_file_path}")
        shutil.copy2(src_file_path, dst_file_path)
            
            
def noAnnotationImgGenerateBox(datset_set_root):
    imgPaths = []
    getExplicitPostfixFiles(datset_set_root, imgPaths)
    for whole_img_path in imgPaths:
        w_index = 0
        image = cv2.imdecode(np.fromfile(whole_img_path, dtype=np.uint8), -1)               # 读图
        if image is None:
            continue
        img_size = image.shape[:2]

        base_name = os.path.basename(whole_img_path)
        base_name = base_name[:base_name.rfind(".")]                                        # 获取纯净的文件名字, 方便拼接
        random_boxes = generateBox(img_size, rand_range_wh=(50, 500), box_num=1)           # 随机生成Box

        for box in random_boxes:
            if box is None:                                                                 # Box 过滤: 1.为空; 2.lable 非法
                continue
            save_file_name = f"{base_name}_{w_index}"                                       # 保存文件纯净文件名字构造
            w_index += 1
            whole_crop_save_path = os.path.join(save_dir, save_file_name) + ".jpg"   # 保存图像文件的全路径构造
            writeBox(image, box, whole_crop_save_path) 


if __name__ == "__main__":
    val_ratio = 0.1
    train_ratio = 0.9
    scale = 0.2

    dataset_name = "/wangkaixiong/dataset/five_class_final_all_dataset_exp_ADD_MASK"                                          # 创建 train\val
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    train_path = f"{dataset_name}/train"                     
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    val_path = f"{dataset_name}/val"
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    dataset_paths = [
        "/FrontierDefensePicture/Image/Clean/无目标数据/监控数据",
    ]
    

    save_dir = "/wangkaixiong/dataset/5class_含动物/Mask"                                          # 创建 train\val
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for dataset_path in dataset_paths:
        noAnnotationImgGenerateBox(dataset_path)
        print(f"dataset: {dataset_path}")
        # getDatasetInfo(dataset_path)
        # split_dataset(save_dir, val_ratio, train_ratio)