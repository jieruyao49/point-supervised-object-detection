import numpy as np
import os
from PIL import Image,ImageDraw,ImageFont
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

classes = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# define color
red = (255,0,0)  # pedestrain  √
yellow = (255,255,0)  # person  √
green = (0,255,0)  # bicycle  √
pink = (255,192,203)  # car  √
purple = (160,32,240)  # van  √
blue = (0,0,255)  # truck  √
orange = (255,97,0)  # tricycle  √
cyan = (0,255,255)  # awning-tricycle  √
strawberry = (135,38,87)  # bus  √
brown = (128,42,42)  # motor  √
white = (255,255,255) # 11:others  0:ignored
colors = {"pedestrian":red,
          "people":yellow,
          "bicycle":green,
          "car":pink,
          "van":purple,
          "truck":blue,
          "tricycle":orange,
          "awning-tricycle":cyan,
          "bus":strawberry,
          "motor":brown}
root_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/data/VOCdevkit2007/VOC2007'
annotation_path = os.path.join(root_path, 'Annotations_txt')
save_path = '/home/rubyyao/PycharmProjects/MELM/pytorch_MELM-master-yang/draw_proposal'
if not os.path.exists(save_path):
    os.mkdir(save_path)
img_path = os.path.join(root_path, 'JPEGImages')
imgs = os.listdir(img_path)
myfont = ImageFont.truetype('/home/rubyyao/data/fangzhengzhenghei/FZZCHJW.TTF', 10)
for num, img in enumerate(imgs):
    print(num, img)
    img_file = os.path.join(img_path, img)
    save_img = os.path.join(save_path, img)
    ori_img = Image.open(img_file)  # original image
    width,height = ori_img.size
    pic = Image.open(img_file)  # draw in original image
    draw = ImageDraw.Draw(pic)
    anno_file = os.path.join(annotation_path, (img.strip('.jpg')+'.xml'))
    annos = open(anno_file).readlines()
    for lines in annos:
        anno = lines.strip('\n').split(',')
        x0 = int(float(anno[0]))
        x1 = int(float(anno[0]) + float(anno[2]))
        y0 = int(float(anno[1]))
        y1 = int(float(anno[1]) + float(anno[3]))
        color = colors[classes[int(anno[5])-1]]
        draw.line([(x0, y0), (x0, y1)], fill=color, width=2)
        draw.line([(x0, y1), (x1, y1)], fill=color, width=2)
        draw.line([(x1, y1), (x1, y0)], fill=color, width=2)
        draw.line([(x1, y0), (x0, y0)], fill=color, width=2)
        # write cls and xsls
        text_coordinate_x = x0 + int(float(anno[2]))/2
        text_coordinate_y = y1 + 3
        text_coordinate_y_xsls = text_coordinate_y + 10
        draw.text((text_coordinate_x, text_coordinate_y), anno[5], font=myfont, fill=color, width=3) # cls
        draw.text((text_coordinate_x, text_coordinate_y_xsls), anno[0], font=myfont, fill=color, width=3) # cls
    joint_img = Image.new(ori_img.mode, (2*width, height))
    joint_img.paste(ori_img, box=(0, 0))
    joint_img.paste(pic, box=(width, 0))
    joint_img.save(save_img)
print('all done')