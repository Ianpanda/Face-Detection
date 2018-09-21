# coding:utf-8 #
# author: Ianpanda
# GitHub: https://github.com/Ianpanda/Face-Detection

import sys, random, cv2
sys.path.append('tools')
from detector import *

def load_mtcnn_model():
    import tensorflow as tf
    import detect_mtcnn as mtcnn
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)
    return pnet, rnet, onet

def randomColor(top=0, bottom=230):
    top = int(top)
    bottom = int(bottom)
    r = random.randint(top + 10, bottom)
    g = random.randint(top, bottom - 30)
    b = random.randint(top, bottom - 30)
    return r,g,b

def show_detector_labels(img, chooses, colors):
    path = img.shape[0] // 20
    label_y = path
    if not isinstance(colors, list):
        colors = [colors]
    for i, color in enumerate(colors):
        cv2.putText(img, chooses[i], (img.shape[0]-path, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        label_y += path
    return img

def drawFaces(image, choose="haar", color=None, pnet=None, rnet=None, onet=None):
    image_ori = image.copy()
    color = randomColor() if not color else color
    if choose == "haar":
        faces = detectFaces_haar(image)
    elif choose == "dlib":
        faces = detectFaces_dlib(image)
    elif choose == "mtcnn":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detectFaces_mtcnn(image, pnet, rnet, onet)
    else:
        raise TypeError('Invalid choose for detector', choose)
    if faces:
        for (x1,y1,x2,y2) in faces:
            x1 = int(x1); x2 = int(x2); y1 = int(y1); y2 = int(y2)
            cv2.rectangle(image_ori, (x1, y1), (x2, y2), color, 2)
    return image_ori, color