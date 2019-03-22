import cv2
import numpy as np
import os
import dlib
from imutils import face_utils

face_det = dlib.get_frontal_face_detector()
marks_det = dlib.shape_predictor("./utils/shape_predictor_68_face_landmarks.dat")

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(face, (96, 96), interpolation=cv2.INTER_LINEAR)/255.

def align_faces(frame, gray, rect):
    shape = marks_det(gray, rect)
    shape = face_utils.shape_to_np(shape).astype(float)
    lefteye = shape[36:42].mean(axis=0).astype("int")
    righteye = shape[42:48].mean(axis=0).astype("int")
    centreye = (lefteye+righteye)//2

    dy, dx = righteye[1]-lefteye[1], righteye[0]-lefteye[0]
    angle = np.degrees(np.arctan2(dy,dx))
    w, h = rect.width()+20, rect.height()+20
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desiredDist = 0.5*w
    scale = desiredDist / dist
    M = cv2.getRotationMatrix2D(tuple(centreye), angle, scale)
    M[0, 2] += (w* 0.5 - centreye[0])
    M[1, 2] += (h* 0.25 - centreye[1])
    outputImg = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    return outputImg

def fetch_face_coords(gray):
    faces = face_det(gray, 0)
    return faces

def get_from_cam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC, 50)
    coords = None
    while True:
        ret, frame = cap.read()
        img = np.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coords = fetch_face_coords(frame)
        if len(coords)!=0:
            rect = coords[0]
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,0,0),2)
        else:
            cv2.putText(frame, 'No face detected!', (100,400), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)
        cv2.putText(frame, "press 'q' to quit!", (100,440), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,0), 2)
        cv2.imshow('face_detect', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    if len(coords)!=0:
        return align_faces(img, gray, rect)
    return None

def get_from_image():
    filename = raw_input("Enter filename/path : ")
    if not os.path.isfile(filename):
        print "No such file"
        return None
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (640, int(640.*frame.shape[0]/frame.shape[1])))
    img = np.copy(frame)
    coords = fetch_face_coords(frame)
    if len(coords)!=0:
        rect = coords[0]
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame,(x, y), (x+w, y+h),(255,0,0),2)
    else:
        cv2.putText(frame, 'No face detected!', (100,400), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)
    cv2.putText(frame, "press any key to quit!", (100,440), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,0,0), 2)
    cv2.imshow('face_detect', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(coords)!=0:
        return align_faces(img, gray, rect)
    return None

def get_face(ch=0):
    # mode 0 : camera
    if ch==0: return get_from_cam()
    # mode 1: image
    elif ch==1: return get_from_image()

def return_face(mode=0):
    c = 'y'
    while c=='y':
        face = get_face(mode)
        if face is not None:
            return preprocess_face(face)
        c = raw_input("No face was detected, want to try again?([y]/n) : ").strip().lower()
    return None
