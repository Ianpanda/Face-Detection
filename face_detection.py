# coding:utf-8 #
# author: Ianpanda
# GitHub: https://github.com/Ianpanda/Face-Detection

import os
import sys
import cv2
import argparse
from tools import utils

def video_process(filename_in, filename_out, detector_choose, speed_up, show):
    if filename_in == 0:
        file_ext = 'system camera:0'
    else:
        file_ext = os.path.splitext(filename_in)[1]
    if file_ext.lower().lstrip('.') in ['jpg','jpeg','png','bmp']:
        filename_out = "".join([filename_out, file_ext])
        frame = cv2.imread(filename_in)
        if detector_choose == "haar":
            detect_img = utils.drawFaces(frame)
        elif detector_choose == "dlib":
            detect_img = utils.drawFaces(frame, "dlib")
        elif detector_choose == "all":
            detect_img = utils.drawFaces(frame)
            detect_img = utils.drawFaces(detect_img, "dlib")
        if show:
            cv2.imshow("detect", detect_img)
        cv2.imwrite(filename_out, detect_img)
        cv2.imshow("image_original", frame)
        cv2.waitKey()
    else:
        filename_out = "".join([filename_out, '.avi'])
        try:
            video_read = cv2.VideoCapture(filename_in)
            fps = video_read.get(cv2.CAP_PROP_FPS)
            size = (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_setup = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            video_save = cv2.VideoWriter(filename_out, video_setup, fps * speed_up, size)
        except SystemError:
            raise TypeError("unsupportable input file", filename_in)
        else:
            while (video_read.isOpened()):
                ret, frame = video_read.read()
                if ret == True:
                    if detector_choose == "haar":
                        detect_img = utils.drawFaces(frame)
                    elif detector_choose == "dlib":
                        detect_img = utils.drawFaces(frame, "dlib")
                    elif detector_choose == "all":
                        detect_img = utils.drawFaces(frame)
                        detect_img = utils.drawFaces(detect_img, "dlib")
                    if show:
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

def command():
    parse = argparse.ArgumentParser(description="detect faces based on haar or dlib",
                                     epilog="demo：python face_detection.py -i file_in -o file_out -d detector\n"
                                            "1.-i:video or image to detect\n"
                                            "2.-o:result\n"
                                            "3.-d:choose your detector\n"
                                            "4.-s:spped up result if need\n"
                                            "5.-v:visualize the detect result immediately",
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parse.add_argument('-i','--input',help="video or image to detect",default=0)
    parse.add_argument('-o','--output',help="video or image result",type=str,default="result")
    parse.add_argument('-d','--detector',help="choose your detector",type=str,choices=['haar','dlib','all'],default='haar')
    parse.add_argument('-s','--speed',help="spped up result if need",type=int,default=1)
    parse.add_argument('-v','--visualize',help="visualize the detect result immediately",type=bool,default=True)
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = command()
    video_process(args.input, args.output, args.detector, args.speed, args.visualize)

