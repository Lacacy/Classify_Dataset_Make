# _*_ coding:utf-8 _*_

from tools.nn_utils import iou
from custom_data_prcoess.label_mapping import label_map
import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
from random import shuffle
import yaml
from utils_tools import read_xml
from custom_data_prcoess.rider_move import clean_rider


def get_xml_size(per_xml_path):
    """
    :param per_xml_path: 具体的xml文件路径
    :return: xml_size(w, h) :int
    """
    tree = ET.parse(per_xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return 0, 0

    if size.find('width') is None:
        return 0, 0
    if size.find('height') is None:
        return 0, 0

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


def bIsBoxIndepend(random_box, choosed_boxes):
    if choosed_boxes is None or random_box is None:
        return False

    for c_box in choosed_boxes:
        if c_box is not None:
            ratio = iou(random_box, c_box[:4])
            # print("ratio: ", ratio)
            if ratio > 0.0:
                return False
    return True

# TODO: load_voc_annotation 考虑优化
# TODO: 增加 考虑优化 flit_logi
# TODO: 类之中考虑增加一个方法, 用来从其他地方生成Mask ;


class GeneratorClassifyFromAnnotations:
    """
    这类做的事情如下：
    1. 指定输出类别; class_name_dict
    2. 数据集的 label_path
    3. 最终保存的目录 dst_root

    source:
        dataset
            Label
            Image
    destination:
        dst_root
            "len(class_name_dict)"class_os.path.basename(label_path)
                class0
                class1
                class2
                class3
                ...
                Mask

    做的核心事情:
    1. 通过 xml文件的标注信息 以及 label_map 获得指定类别的 box,
        - 并且采用自定义逻辑对类别以及框的位置进行校正(骑手过长的box缩短一部分);
        - 采用类内部定义的box宽高过滤逻辑进行小目标过滤;
    2. 通过 xml文件的标注信息 以及 label_map 获得指定类别的 box,
        - 随机生成 Box;
        - 随机生成的 Box和指定类别Box做IOU, 只保留没有交集的Box作为 Mask类别;
    3. 过 xml文件的标注信息 以及 label_map 获得指定类别的 box,
        - 指定类别的 Box映射为 Mask;
        - crop -> save

    """
    def __init__(self, label_path, dst_root, class_name_dict, choose_data_num=20000,
                img_postfix=[".jpeg", ".jpg", ".png", ".jfif", ".bmp", ".tiff", ".JPEG", ".Jpeg"], annotations_flit_func=None):
        self.label_dir = label_path
        self.img_postfix = img_postfix
        self.class_name_dict = class_name_dict
        self.num_class = len(self.class_name_dict)
        token = os.path.basename(self.label_dir)
        self.save_root = os.path.join(dst_root, f"{self.num_class}class_{token}")
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        self.choose_data_num = choose_data_num
        self.flit_func = None
        self.save_postfix = ".jpg"
        if annotations_flit_func is not None:
            self.flit_func = annotations_flit_func

    def writeBox(self, img, box, save_path):
        if img is not None:
            save_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # print("save_path", save_path)
            cv2.imwrite(save_path, save_img)

    def checkImgExist(self, whole_img_path):
        """
        自动去匹配路径;
        返回值:
            不存在, False, None
            存在,   True, ImgPath
        """
        img_file_flag = False
        for postfix in self.img_postfix:
            whole_img_path = whole_img_path[:whole_img_path.rfind(".")] + postfix  # 切换成图像文件的后缀形式
            if os.path.exists(whole_img_path):
                img_file_flag = True
                break
            else:
                continue

        if not img_file_flag:
            return False, None

        return True, whole_img_path

    def generate_from_labelmap(self):
        xml_paths = os.listdir(self.label_dir)
        data_num = min(len(xml_paths), self.choose_data_num)
        shuffle(xml_paths)
        xml_paths = xml_paths[:data_num]
        flit_xml_files = list(filter(lambda x: x[x.rfind(".") + 1:] == "xml", xml_paths))  # 过滤后缀
        for xml_name in flit_xml_files:
            whole_xml_path = os.path.join(self.label_dir, xml_name)  # xml 全路径拼接
            # print(whole_xml_path)
            whole_img_path = whole_xml_path.replace("Label", "Image")  # ! TODO: 目录切换到图像文件所在目录
            img_file_flag = False
            # TODO 写成一个函数
            ret, whole_img_path = self.checkImgExist(whole_img_path)
            if not ret:
                continue

            # print("whole_img_path", whole_img_path)
            image = cv2.imdecode(np.fromfile(whole_img_path, dtype=np.uint8), -1)  # 读图
            if image is None:
                continue

            base_name = os.path.basename(whole_img_path)
            base_name = base_name[:base_name.rfind(".")]  # 获取纯净的文件名字, 方便拼接
            boxes = read_xml(whole_xml_path, label_map)  # vocXml2Box
            if isinstance(boxes, list):
                if len(boxes) == 0:
                    continue
            elif isinstance(boxes, np.ndarray):
                if boxes.shape[0] == 0:
                    continue

            if self.flit_func is not None:
                print(f"flit_func is not none")
                self.flit_func(boxes, [0, 8], [2])  # TODO: 等待实现
            else:
                print(f"flit_func is None")

            self.flit_box_by_w_h(boxes)
            boxes = list(filter(lambda x: x[4] != -1, boxes))

            w_index = 0
            for box in boxes:
                if box is None:  # Box 过滤: 1.为空; 2.lable 非法
                    continue
                label = int(box[4])

                save_file_name = f"{base_name}_{w_index}"  # 保存文件纯净文件名字构造
                w_index += 1
                save_dir = os.path.join(self.save_root, self.class_name_dict[label])  # 保存文件纯净文件的子目录构造
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                whole_crop_save_path = os.path.join(save_dir, save_file_name) + self.save_postfix  # 保存图像文件的全路径构造
                self.writeBox(image, box, whole_crop_save_path)  # 写 crop 的小图到指定路径之中

    def make_background_box(self, num=8000, bg_name="Mask"):
        xml_paths = os.listdir(self.label_dir)
        shuffle(xml_paths)
        flit_xml_files = list(filter(lambda x: x[x.rfind(".") + 1:] == "xml", xml_paths))  # 过滤后缀
        bg_count = 0
        for xml_name in flit_xml_files:
            if bg_count > num:
                break
            whole_xml_path = os.path.join(self.label_dir, xml_name)  # xml 全路径拼接
            # print(whole_xml_path)
            whole_img_path = whole_xml_path.replace("Label", "Image")  # ! TODO: 目录切换到图像文件所在目录
            ret, whole_img_path = self.checkImgExist(whole_img_path)
            if not ret:
                continue

            # print("whole_img_path", whole_img_path)
            image = cv2.imdecode(np.fromfile(whole_img_path, dtype=np.uint8), -1)  # 读图
            if image is None:
                continue

            base_name = os.path.basename(whole_img_path)
            base_name = base_name[:base_name.rfind(".")]                              # 获取纯净的文件名字, 方便拼接
            img_size = get_xml_size(whole_xml_path)                                   # 获取xml文件中记录的图像高度
            random_boxes = generateBox(img_size, rand_range_wh=(50, 500), box_num=7)  # 随机生成Box
            boxes = read_xml(whole_xml_path, label_map)  # vocXml2Box
            if isinstance(boxes, list):
                if len(boxes) == 0:
                    continue
            elif isinstance(boxes, np.ndarray):
                if boxes.shape[0] == 0:
                    continue

            w_index = 0
            for r_box in random_boxes:                                  # 随机生成Box与GT做IOU, 写入Mask文件夹
                if bIsBoxIndepend(r_box, boxes):
                    save_file_name = f"{base_name}_{w_index}"
                    w_index += 1
                    save_dir = os.path.join(self.save_root, bg_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    whole_crop_save_path = os.path.join(save_dir, save_file_name) + self.save_postfix
                    self.writeBox(image, r_box, whole_crop_save_path)
                    bg_count += 1
                    r_box.append(-1)
                    boxes = np.append(boxes, [r_box], axis=0)


    def flit_box_by_w_h(self, boxes):
        for item in boxes:
            w = item[2] - item[0]
            h = item[3] - item[1]
            if w < 60 or h < 60:
                item[4] = -1





if __name__ == "__main__":

    cfg_file = "data_config/dataset_info.yaml"
    with open(cfg_file, encoding='UTF-8') as f:
        file_stream = yaml.load(f, yaml.FullLoader)
    xml_lable_dataset = file_stream["data_list"]
    dst_root = file_stream["subDataset_save_root"]
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    num_class = file_stream["class_num"]
    save_class = {0: "Person", 1: "Veh", 2: "NoVeh", 3: "Head"}
    print(f"xml_label_dataset: {xml_lable_dataset}, dst_root: {dst_root}")
    # exit()
    for datasetLabel in xml_lable_dataset:
        engine = GeneratorClassifyFromAnnotations(datasetLabel, dst_root, save_class,
            choose_data_num=8000, annotations_flit_func=clean_rider)
        engine.generate_from_labelmap()
        engine.make_background_box(num=10000, bg_name="Mask")