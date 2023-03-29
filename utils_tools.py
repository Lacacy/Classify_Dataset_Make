
import os
import xml.etree.ElementTree as ET


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


def load_annotation(annotation_file, label_dict):
    '''
    加载标注文件xml，读取其中的bboxes
    参数：
        annotation_file[str]：  指定为xml文件路径
        label_map[list]：       指定为标签数组
    返回值：
        np.array([(xmin, ymin, xmax, ymax, class_index), (xmin, ymin, xmax, ymax, class_index)])
    '''
    with open(annotation_file, "r", encoding="UTF-8") as f:
        annotation_data = f.read()

    def middle(s, begin, end, pos_begin=0):
        p = s.find(begin, pos_begin)
        if p == -1:
            return None, None

        p += len(begin)
        e = s.find(end, p)
        if e == -1:
            return None, None

        return s[p:e], e + len(end)

    obj_bboxes = []
    object_, pos_ = middle(annotation_data, "<object>", "</object>")
    while object_ is not None:
        xmin = float(middle(object_, "<xmin>", "</xmin>")[0])
        ymin = float(middle(object_, "<ymin>", "</ymin>")[0])
        xmax = float(middle(object_, "<xmax>", "</xmax>")[0])
        ymax = float(middle(object_, "<ymax>", "</ymax>")[0])
        name = middle(object_, "<name>", "</name>")[0]
        cls = label_dict(name.strip())
        object_, pos_ = middle(annotation_data, "<object>", "</object>", pos_)
        obj_bboxes.append([xmin, ymin, xmax, ymax, cls])

    return obj_bboxes


def read_xml(single_xml_path, label_dict):
    box_list = []
    if not os.path.exists(single_xml_path):
        return box_list
    tree = ET.parse(single_xml_path)
    root = tree.getroot()
    #height = 0
    #width = 0
    #for s in root.iter('size'):
    #    height = int(s.find('height').text)
    #    width = int(s.find('width').text)
    for obj in root.iter('object'):
        # print(obj.find('name').text)
        cls = label_dict(obj.find('name').text)
        xmlbox = obj.find('bndbox')
        box = [float(xmlbox.find(x).text) for x in ('xmin', 'ymin', 'xmax', 'ymax')]
        box.append(cls)
        box_list.append(box)
    return box_list


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