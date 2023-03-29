import sys

# sys.path.append("..")
from tools.nn_utils import iou

def custom_key(item):
    return item[0]  # second parameter denotes the age

def old_clean_rider(Boxes: list, person_cls: list, noveh_cls: list):
    """
    1. 获取行人、骑手的索引
    2. 每一个人和所有非机动车做 BoxIOU，
    3. 符合条件的行人、骑手, 缩回到非机动车的一半; 并且更改类别为行人
    """
    person_indices = []
    for index, box in enumerate(Boxes):
        if box[4] in person_cls:
            person_indices.append(index)

    for ip in person_indices:
        ratio_list = []
        for index, box in enumerate(Boxes):
            if box[4] in noveh_cls:
                ratio = iou(Boxes[ip][:4], box[:4])
                item = (ratio, index)
                ratio_list.append(item)

        if len(ratio_list) != 0:
            ratio_list.sort(key=custom_key, reverse=True)
            # print(ratio_list)
            # ratio_list[0][1]: 最高得分: 0, 最高得分对应的索引
            box = Boxes[ratio_list[0][1]]
            max_ratio = ratio_list[0][0]
            #                  bottom - up = height
            box_half_height = (box[3] - box[1]) * 0.5
            box_half_height_limit = (box[3] - box[1]) * 0.65
            # 非预估骑手框, 即 手工标注的骑手半身框, 不做处理
            if Boxes[ip][3] < (box[1] + box_half_height_limit):
                continue

            if max_ratio > 0.3 and Boxes[ip][3] > (box[1] + box_half_height):
                # print(max_ratio)
                # person: bottom
                Boxes[ip][3] = box[1] + box_half_height
                Boxes[ip][4] = person_cls[1]


def clean_rider(Boxes: list, person_cls: list, noveh_cls: list):
    """
    1. 获取行人、骑手的索引
    2. 每一个人和所有非机动车做 BoxIOU，
    3. 符合条件的行人、骑手, 缩回到非机动车的一半; 并且更改类别为行人
    """
    # print(Boxes)
    person_indices = []
    for index, box in enumerate(Boxes):
        if box[4] in person_cls:
            person_indices.append(index)

    for ip in person_indices:
        ratio_list = []
        for index, box in enumerate(Boxes):
            if box[4] in noveh_cls:
                ratio = iou(Boxes[ip][:4], box[:4])
                item = (ratio, index)
                ratio_list.append(item)

        if len(ratio_list) != 0:
            ratio_list.sort(key=custom_key, reverse=True)
            # print(ratio_list)
            # ratio_list[0][1]: 最高得分: 0, 最高得分对应的索引
            box = Boxes[ratio_list[0][1]]
            max_ratio = ratio_list[0][0]
            #                  bottom - up = height
            rider_height_limit = (box[3] - box[1]) * 0.6
            person_height = Boxes[ip][3] - Boxes[ip][1]
            person_bottom = Boxes[ip][3]
            box_half_height_limit = (box[3] - box[1]) * 0.65
            # 非预估骑手框, 即 手工标注的骑手半身框, 不做处理
            if person_bottom < (box[1] + box_half_height_limit):
                continue

            if max_ratio > 0.3 and person_height > rider_height_limit:
                # print(max_ratio)
                # person: bottom
                Boxes[ip][3] = box[1] + person_height * 0.75
                Boxes[ip][4] = person_cls[0]