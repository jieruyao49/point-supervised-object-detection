import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

classes = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
eachcls_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/results/VOC2007/Main'
txt = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
save_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/results/VOC2007/Main/image_format'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for index, cls in enumerate(classes):
    print('{} {}'.format(index, cls))
    img_lines = open(os.path.join(eachcls_path, cls+'.txt')).readlines()
    for line in img_lines:
        eles = line.strip('\n').split(' ')
        write = open(os.path.join(save_path, eles[0]+'.txt'), 'a')
        write_info = '{} {} {} {} {} {}\n'.format(index, eles[1], eles[2], eles[3], eles[4], eles[5])
        write.write(write_info)
        write.close()