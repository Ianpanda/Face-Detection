# Face-Detection

Face detection based on haar, dlib, and mtcnn detectors

[![GPL Licence](https://badges.frapsoft.com/os/gpl/gpl.svg?v=103)](https://opensource.org/licenses/GPL-3.0/)

## Introduction

A simple implementation of face detection (Dlib/OpenCV/MTCNN backend)

---

## Quick Start

1. Clone this repo.
2. Run face detection.

```
git clone https://github.com/Ianpanda/Face-Detection.git
python face_detection.py -i demo_images/00.jpg for image detection mode, OR
python face_detection.py -i [video_path] -o [output_path (optional)] for video detection mode.
```

## Usage

Use -h or --help to see usage of face_detection.py:

```
usage: face_detection.py [-h] [-i INPUT] [-o OUTPUT]
                         [-d {haar,dlib,mtcnn,all}] [-s SPEED] [-v {y,n}]

detect faces based on haar or dlib

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        video or image to detect
  -o OUTPUT, --output OUTPUT
                        video or image result
  -d {haar,dlib,mtcnn,all}, --detector {haar,dlib,mtcnn,all}
                        choose your detector
  -s SPEED, --speed SPEED
                        spped up video result if need
  -v {y,n}, --visualize {y,n}
                        visualize the detect result immediately

demo：python face_detection.py -i file_in -o file_out -d detector
1.-i:video or image to detect
2.-o:result
3.-d:choose your detector
4.-s:spped up video result if need
5.-v:visualize the detect result immediately
```

---

## Some issues to know

1. The test environment is
    - Python 3.5.4
    - OpenCV 3.3.1
    - Dlib 19.15.0
	- TensorFlow 1.9.0

2. Default haar classifier are used. If you use others, try to get from $ROOT/model_weights/haar, or you can train your own xml.

3. The MTCNN model has fixed weights in model_weights/mtcnn.
---

## To Do List

- [ ] Update method to train xml
- [x] Add MTCNN detector
- [ ] Extend other amazing Networks

