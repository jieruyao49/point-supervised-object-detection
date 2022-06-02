import xml.etree.ElementTree as ET
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_xml(file):
    tree = ET.parse(file)
    return tree


def iter_xml(root):
    for node in list(root):
        print(node)


classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')


xml_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/Annotations/'
txt_save_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/Annotations_txt/'
txt_test_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'

txt = open(txt_test_path)
lines = txt.read().splitlines()
# order = np.array([0, 3, 2, 1])

for line in lines:

    xml_name = xml_path + line + '.xml'
    txt_name = txt_save_path + line + '.txt'
    file = open(txt_name, 'w')

    tree = read_xml(xml_name)
    root = tree.getroot()

    i = 0
    for ob in root.iter('object'):
        for name in ob.iter('name'):
            result = name.text in classes
            if result == True:
                input_name = name.text + ' '
                file.write(input_name)
            # print('name ', name.text)

        # for pose in ob.iter('pose'):
        #     print('pose ', pose.text)
        # for trun in ob.iter('truncated'):
        #     print('trun ', trun.text)
        # for diff in ob.iter('difficult'):
        #     print('diff ', diff.text)
        for bndbox in ob.iter('bndbox'):
            bbox = np.array([0, 0, 0, 0])
            j = 0
            for l in bndbox:
                if str.isdigit(l.text) == False:
                    break

                size = int(l.text)
                bbox[j] = size
                j = j + 1

            if (bbox == np.array([0, 0, 0, 0])).all() == True:
                continue

            for j in range(4):
                size = str(bbox[j])
                if j == 3:
                    size = size + '\n'
                else:
                    size = size + ' '

                file.write(size)

    file.close()
    print(line)