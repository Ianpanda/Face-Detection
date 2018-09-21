from scipy import misc
import tensorflow as tf
import os, sys, cv2
import matplotlib.pyplot as plt
from tools import utils
from tools import detect_mtcnn as mtcnn


def img_process(filename_in, filename_out=None, model=None):
    if not filename_out:
        file_ext = os.path.splitext(filename_in)[1]
        filename_out = filename_in.replace(file_ext, '_detect'+file_ext)
    if not model:
        model = utils.load_mtcnn_model()
    pnet, rnet, onet = model
    img = cv2.imread(filename_in)
    cv2.imshow("img_original", img)
    detect_img, color = utils.drawFaces(img, choose="mtcnn", pnet=pnet, rnet=rnet, onet=onet)
    detect_img = utils.show_detector_labels(detect_img, ["mtcnn"], color)
    cv2.imwrite(filename_out, detect_img)
    cv2.imshow("detect", detect_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    file_in = "demo_images/02.jpg"
    img_process(file_in)