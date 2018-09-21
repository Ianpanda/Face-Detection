# coding:utf-8 #
# author: Ianpanda
# GitHub: https://github.com/Ianpanda/Face-Detection


# function {detectFaces} returns the bounding box of faces as format (lefttop, rightbottom)
def detectFaces_haar(image):
    import cv2
    # use haar classifier，or you can change it in "$ROOT/model_weights/haar/"
    xml_path = "model_weights/haar/haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(xml_path)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    faces = face_cascade.detectMultiScale(gray, 1.2, 5) # 1.2 and 5 for min and max windows,changes will result in different detects
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

def detectFaces_dlib(image):
    import dlib
    detector = dlib.get_frontal_face_detector()
    detect = detector(image)
    result = []
    for point in detect:
        x1 = point.left()
        y1 = point.top()
        x2 = point.right()
        y2 = point.bottom()
        result.append((x1, y1, x2, y2))
    return result

def detectFaces_mtcnn(image, pnet, rnet, onet):
    import detect_mtcnn as mtcnn
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    bounding_boxes, _ = mtcnn.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    bounding_boxes = bounding_boxes[:,:4]
    return list(bounding_boxes)

