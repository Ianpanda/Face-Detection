# coding:utf-8 #
# author: Ianpanda
# GitHub: https://github.com/Ianpanda/Face-Detection

import random
from tools.detector import *

def randomColor(top=0, bottom=230):
    top = int(top)
    bottom = int(bottom)
    r = random.randint(top + 10, bottom)
    g = random.randint(top, bottom - 30)
    b = random.randint(top, bottom - 30)
    return r,g,b

def drawFaces(image, choose="haar"):
    image_ori = image.copy()
    faces = None
    color = randomColor()
    if choose == "haar":
        faces = detectFaces_haar(image)
    elif choose == "dlib":
        faces = detectFaces_dlib(image)
        color = randomColor(bottom=200)
    else:
        raise TypeError('Invalid choose for detector', choose)
    if faces:
        for (x1,y1,x2,y2) in faces:
            cv2.rectangle(image_ori, (x1, y1), (x2, y2), color, 2)
    return image_ori