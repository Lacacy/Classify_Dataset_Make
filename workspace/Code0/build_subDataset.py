# _*_ coding:utf-8 _*_
from tools.nn_utils import load_voc_annotation, iou
from label_mapping import label_map
import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
from random import shuffle
import yaml

# a. 解析 xml 读取成 List[[Box]
# b. 根据类别裁剪数据；生成
# 	i. 机动车
# 	ii. 非机动车
# 	iii. 行人
# 	iv. 头肩
# 	v. Mask：
# 		1) 利用 nn_utils 的 解析每张图的 xml 生成 ListBox
# 		2) 每张图生成随机15个box；  代码基本实现
# 		3) 利用 nn_utils 的 15个随机box 依次与 ListBox 做IOU过滤，没被过滤掉则加入到背景类之中；没被过滤掉的ListBox.append(saveRandomBox)
# 			a) list(fliter(func， container)
# 			b) 遍历
#         裁剪数据

# label: -1,表示数据集之中的背景;
# 1: 表示行人
# 2: 表示机动车
# 3: 非机动车
# 4: 头肩
# 5: 人脸

def get_xml_size(per_xml_path):
    """
    :param per_xml_path: 具体的xml文件路径
    :return: xml_size(w, h) :int
    """
    tree = ET.parse(per_xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return 0

    if size.find('width') is None:
        return 0
    if size.find('height') is None:
        return 0

    w = int(size.find('width').text)
    h = int(size.find('height').text)
    return h, w


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

def bIsBoxIndepend(random_box, choosed_boxes):
    if choosed_boxes is None or random_box is None:
        return False
    
    if choosed_boxes.shape[0] == 0:
        return False

    for c_box in choosed_boxes:
        if c_box is not None:
            ratio = iou(random_box, c_box[:4])
            # print("ratio: ", ratio)
            if ratio > 0.0:
                return False
    return True


def doSomething(dataSetLablePath):
    xmls_dir = [dataSetLablePath]
    token = os.path.basename(dataSetLablePath)
    save_root = f"{dst_root}/{class_token}{token}"
    save_class = {1: "Person", 2: "Veh", 3: "NoVeh", 4: "Head"}
    image_postfix = [".jpeg", ".jpg", ".png", ".jfif", ".bmp", ".tiff", ".JPEG", ".Jpeg"]
    save_postfix = ".jpg"
    need_make_dir = "Mask"
    data_num = 5000

    for xml_dir in xmls_dir:
        xml_files = os.listdir(xml_dir)                                                         # 获取子xml的文件夹
        data_num = min(len(xml_files), data_num)
        xml_files = xml_files[:data_num]
        shuffle(xml_files)
        filt_xml_files = list(filter(lambda x: x[x.rfind(".") + 1:] == "xml", xml_files))       # 过滤后缀
        # print(filt_xml_files)
        for xml_file in filt_xml_files:                                                         # 遍历所有xml文件
            whole_xml_path = os.path.join(xml_dir, xml_file)                                    # xml 全路径拼接
            # print(whole_xml_path)
            whole_img_path = whole_xml_path.replace("Label", "Image")                           #! TODO: 目录切换到图像文件所在目录
            img_file_flag = False
            for postfix in image_postfix:
                whole_img_path = whole_img_path[:whole_img_path.rfind(".")] + postfix         # 切换成图像文件的后缀形式
                if os.path.exists(whole_img_path):
                    img_file_flag = True
                    break
                else:
                    continue
                    
            if not img_file_flag:
                continue
            
            # print("whole_img_path", whole_img_path)
            image = cv2.imdecode(np.fromfile(whole_img_path, dtype=np.uint8), -1)               # 读图
            if image is None:
                continue

            base_name = os.path.basename(whole_img_path)
            base_name = base_name[:base_name.rfind(".")]                                        # 获取纯净的文件名字, 方便拼接
            img_size = get_xml_size(whole_xml_path)                                             # 获取xml文件中记录的图像高度
            random_boxes = generateBox(img_size, rand_range_wh=(50, 500), box_num=7)           # 随机生成Box
            boxes = load_voc_annotation(whole_xml_path, label_map)                              # vocXml2Box
            if boxes.shape[0] == 0:
                continue
                
            w_index = 0
            for box in boxes:
                if box is None:                                                                 # Box 过滤: 1.为空; 2.lable 非法
                    continue
                label = int(box[4])
                if label == -1:
                    continue

                save_file_name = f"{base_name}_{w_index}"                                       # 保存文件纯净文件名字构造
                w_index += 1
                save_dir = os.path.join(save_root, save_class[label])                           # 保存文件纯净文件的子目录构造
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                whole_crop_save_path = os.path.join(save_dir, save_file_name) + save_postfix   # 保存图像文件的全路径构造
                writeBox(image, box, whole_crop_save_path)                                      # 写 crop 的小图到指定路径之中

            for r_box in random_boxes:                                                          # 随机生成Box与GT做IOU, 写入Mask文件夹
                if bIsBoxIndepend(r_box, boxes):
                    save_file_name = f"{base_name}_{w_index}"
                    w_index += 1
                    save_dir = os.path.join(save_root, need_make_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    whole_crop_save_path = os.path.join(save_dir, save_file_name) + save_postfix
                    writeBox(image, r_box, whole_crop_save_path)
                    r_box.append(-1)
                    # boxes = np.add(boxes, r_box)
                    boxes = np.append(boxes, [r_box], axis=0)


if __name__ == "__main__":

    cfg_file = "data_config/dataset_info.yaml"
    with open(cfg_file, encoding='UTF-8') as f:
        file_stream = yaml.load(f, yaml.FullLoader)
    xml_lable_dataset = file_stream["data_list"]
    dst_root = file_stream["subDataset_save_root"]
    num_class = file_stream["class_num"]
    class_token = f"{num_class}class_"
    print(f"xml_label_dataset: {xml_lable_dataset}, dst_root: {dst_root}")
    # exit()
    for datasetLabel in xml_lable_dataset:
        doSomething(datasetLabel)