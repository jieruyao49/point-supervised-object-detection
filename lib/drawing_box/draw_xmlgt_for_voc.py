import xml.etree.ElementTree as ET
import os
from PIL import Image,ImageDraw,ImageFont
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

score_threshold = 0.001 # only draw boxes whose score more than this threshold
green = (0,255,0)
red = (255,0,0)
classes = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

xmlgt_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/Annotations'
results_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/results/VOC2007/Main/image_format'
txt_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
img_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007/JPEGImages'
save_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/output/draw_proposal'

images = open(txt_path).readlines()
if not os.path.exists(save_path):
    os.makedirs(save_path)
for index, img in enumerate(images):
    print(index, img)
    tree = ET.parse(os.path.join(xmlgt_path, img.strip('\n')+'.xml'))
    image = Image.open(os.path.join(img_path, img.strip('\n')+'.jpg'))
    save_file = os.path.join(save_path, img.strip('\n')+'.jpg')
    result_file = os.path.join(results_path, img.strip('\n')+'.txt')
    draw = ImageDraw.Draw(image)
    # draw gt
    objs = tree.findall('object')
    for obj in objs:
        cls = obj.find('name').text.lower().strip()
        cls_index = classes.index(cls)
        box = obj.find('bndbox')
        # draw box
        x0 = int(box.find('xmin').text)
        y0 = int(box.find('ymin').text)
        x1 = int(box.find('xmax').text)
        y1 = int(box.find('ymax').text)
        draw.line([(x0, y0), (x0, y1)], fill=green, width=2)
        draw.line([(x0, y1), (x1, y1)], fill=green, width=2)
        draw.line([(x1, y1), (x1, y0)], fill=green, width=2)
        draw.line([(x1, y0), (x0, y0)], fill=green, width=2)
    # draw results
    results = open(result_file).readlines()
    for result in results:
        eles = result.strip('\n').split(' ')
        if float(eles[1]) >= score_threshold:
            x0 = int(float(eles[2]))
            x1 = int(float(eles[4]))
            y0 = int(float(eles[3]))
            y1 = int(float(eles[5]))
            draw.line([(x0, y0), (x0, y1)], fill=red, width=2)
            draw.line([(x0, y1), (x1, y1)], fill=red, width=2)
            draw.line([(x1, y1), (x1, y0)], fill=red, width=2)
            draw.line([(x1, y0), (x0, y0)], fill=red, width=2)
    image.save(save_file)
print('all done')