# coding:utf-8 #
# author: Ianpanda
# GitHub: https://github.com/Ianpanda/Face-Detection

import os
import sys
import cv2
import argparse
from tools import utils

def video_process(filename_in, filename_out, detector_choose, speed_up, show):
    write_codes = 'MJPG'    # modify the code to suit your system environment, or you can define -1 for manually choose this.
    if filename_in == 0:
        file_ext = 'system camera:0'
    else:
        file_ext = os.path.splitext(filename_in)[1]
    show = True if show == 'y' else False
    if file_ext.lower().lstrip('.') in ['jpg','jpeg','png','bmp']:
        if not filename_out:
            filename_out = filename_in.replace(file_ext, '_detect'+file_ext)
        else:
            filename_out = "".join([filename_out, file_ext])
        frame = cv2.imread(filename_in)
        if detector_choose == "haar":
            detect_img, color = utils.drawFaces(frame)
            colors = [color]
        elif detector_choose == "dlib":
            detect_img, color = utils.drawFaces(frame, "dlib")
            colors = [color]
        elif detector_choose == "mtcnn":
            pnet, rnet, onet = utils.load_mtcnn_model()
            detect_img, color = utils.drawFaces(frame, "mtcnn", pnet, rnet, onet)
            colors = [color]
        elif detector_choose == "all":
            detect_img, _1 = utils.drawFaces(frame)
            detect_img, _2 = utils.drawFaces(detect_img, "dlib", color=utils.randomColor(bottom=200))
            pnet, rnet, onet = utils.load_mtcnn_model()
            detect_img, _3 = utils.drawFaces(detect_img, "mtcnn", utils.randomColor(top=10, bottom=250), pnet, rnet, onet)
            colors = [_1, _2, _3]
        else:
            raise KeyError('Detector {} not yet implemented !'.format(detector_choose))
        detector_chooses = ["haar", "dlib", "mtcnn"] if detector_choose == "all" else [detector_choose]
        detect_img = utils.show_detector_labels(detect_img, detector_chooses, colors)
        if show:
            cv2.imshow("detect", detect_img)
        cv2.imwrite(filename_out, detect_img)
        cv2.imshow("image_original", frame)
        cv2.waitKey()
    else:
        if not filename_out:
            filename_out = "result_{}".format(detector_choose)
        filename_out = "".join([filename_out, '.avi'])
        try:
            video_read = cv2.VideoCapture(filename_in)
            fps = video_read.get(cv2.CAP_PROP_FPS)
            size = (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_setup = cv2.VideoWriter_fourcc(*write_codes)
            video_save = cv2.VideoWriter(filename_out, video_setup, fps * speed_up, size)
        except SystemError:
            raise TypeError("unsupportable input file", filename_in)
        else:
            color_define = []
            for i in range(3):    # the number of supportable detectors
                color_define.append(utils.randomColor())
            if detector_choose in ["mtcnn","all"]:
                pnet, rnet, onet = utils.load_mtcnn_model()
            path = fps // 15
            i = 0
            while (video_read.isOpened()):
                # use path to detect frame(you may get more powerful performance by using key frame instead of path frame)
                ret, frame = video_read.read()
                if ret == True:
                    if i % path == 0:
                        if detector_choose == "haar":
                            detect_img, color = utils.drawFaces(frame, color=color_define[0])
                            colors = [color]
                        elif detector_choose == "dlib":
                            detect_img, color = utils.drawFaces(frame, "dlib", color_define[1])
                            colors = [color]
                        elif detector_choose == "mtcnn":
                            detect_img, color = utils.drawFaces(frame, "mtcnn", color_define[2], pnet, rnet, onet)
                            colors = [color]
                        elif detector_choose == "all":
                            detect_img, _1 = utils.drawFaces(frame, color=color_define[0])
                            detect_img, _2 = utils.drawFaces(detect_img, "dlib", color=color_define[1])
                            detect_img, _3 = utils.drawFaces(detect_img, "mtcnn", color_define[2], pnet, rnet, onet)
                            colors = [_1, _2, _3]
                        else:
                            raise KeyError('Detector {} not yet implemented !'.format(detector_choose))
                        detector_chooses = ["haar", "dlib", "mtcnn"] if detector_choose == "all" else [detector_choose]
                        detect_img = utils.show_detector_labels(detect_img, detector_chooses, colors)
                        if show:
                            cv2.imshow("detect", detect_img)
                        video_save.write(detect_img)
                    else:
                        if show:
                            cv2.imshow("detect", frame)
                        video_save.write(frame)
                    i += 1
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
                                            "4.-s:spped up video result if need\n"
                                            "5.-v:visualize the detect result immediately",
                                     formatter_class=argparse.RawDescriptionHelpFormatter
                                     )
    parse.add_argument('-i','--input',help="video or image to detect",default=0)
    parse.add_argument('-o','--output',help="video or image result",type=str,default=None)
    parse.add_argument('-d','--detector',help="choose your detector",type=str,choices=['haar','dlib','mtcnn','all'],default='haar')
    parse.add_argument('-s','--speed',help="spped up video result if need",type=int,default=1)
    parse.add_argument('-v','--visualize',help="visualize the detect result immediately",choices=['y','n'],type=str,default='y')
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = command()
    video_process(args.input, args.output, args.detector, args.speed, args.visualize)

