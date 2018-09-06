# Face-Detection

Face detection based on haar detector and dlib
[![GPL Licence](https://badges.frapsoft.com/os/gpl/gpl.svg?v=103)](https://opensource.org/licenses/GPL-3.0/)

## Introduction

A simple implementation of face detection (Dlib and OpenCV backend)

---

## Quick Start

1. Clone this repo.
2. Run face detection.

```
git clone https://github.com/Ianpanda/Face-Detection.git
python face_detection.py -i demo.jpg for image detection mode, OR
python face_detection.py -i [video_path] -o [output_path (optional)]
```

## Usage

Use -h or --help to see usage of face_detection.py:

```
usage: face_detection.py [-h] [-i INPUT] [-o OUTPUT] [-d {haar,dlib,all}]
                         [-s SPEED] [-v VISUALIZE]

detect faces based on haar or dlib

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        video or image to detect
  -o OUTPUT, --output OUTPUT
                        video or image result
  -d {haar,dlib,all}, --detector {haar,dlib,all}
                        choose your detector
  -s SPEED, --speed SPEED
                        spped up result if need
  -v VISUALIZE, --visualize VISUALIZE
                        visualize the detect result immediately

demo：python face_detection.py -i file_in -o file_out -d detector
1.-i:video or image to detect
2.-o:result
3.-d:choose your detector
4.-s:spped up result if need
5.-v:visualize the detect result immediately
```

---

## Some issues to know

1. The test environment is
    - Python 3.5.4
    - OpenCV 3.3.1
    - Dlib 19.15.0

2. Default haar classifier are used. If you use others, try to get from $ROOT/tools/haar, or you can train your own xml.


---

## To Do List

- [ ] Update method to train xml
- [ ] Extend other amazing Networks

