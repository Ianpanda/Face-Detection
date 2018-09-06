# coding:utf-8 #
# author: Ianpanda
# GitHub: https://github.com/Ianpanda/Face-Detection

import sys
import cv2
import dlib
from tools import utils

def video_process(filename_in=0, filename_out="result.avi", speed_up=1):
    video_read = cv2.VideoCapture(filename_in)
    fps = video_read.get(cv2.CAP_PROP_FPS)
    size = (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_setup = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video_save = cv2.VideoWriter(filename_out, video_setup, fps*speed_up, size)
    while (video_read.isOpened()):
        ret, frame = video_read.read()

        if ret == True:
            detect_img = utils.drawFaces(frame, choose="dlib")
            cv2.imshow("detect", detect_img)
            video_save.write(detect_img)
            cv2.imshow("video_original", frame)
            if cv2.waitKey(1) >= 0:
                break
        else:
            print("wrong video input!")
            break
    video_read.release()
    video_save.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    file_out = "result_dlib.avi"
    video_process(filename_out=file_out, speed_up=5)