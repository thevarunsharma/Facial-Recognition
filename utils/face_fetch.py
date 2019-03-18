import cv2
import numpy as np
import os

face = cv2.CascadeClassifier()
face.load("/home/beast/opencv/data/haarcascades/haarcascade_frontalface_default.xml")

def fetch_face_coords(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3)
    if len(faces)==0:
        return None
    (x, y, w, h) = faces[0]
    return x, y, w, h

def get_from_cam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC, 50)
    coords = None
    while True:
        ret, frame = cap.read()
        coords = fetch_face_coords(frame)
        if coords:
            x, y, w, h = coords
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
    if coords:
        x, y, w, h = coords
        return frame[y:y+h, x:x+w]
    return None

def get_from_image():
    filename = raw_input("Enter filename/path : ")
    if not os.path.isfile(filename):
        print "No such file"
        return None
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (640, int(640.*frame.shape[0]/frame.shape[1])))
    coords = fetch_face_coords(frame)
    if coords:
        x, y, w, h = coords
        cv2.rectangle(frame,(x, y), (x+w, y+h),(255,0,0),2)
    else:
        cv2.putText(frame, 'No face detected!', (100,400), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)
    cv2.putText(frame, "press any key to quit!", (100,440), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,0,0), 2)
    cv2.imshow('face_detect', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if coords:
        x, y, w, h = coords
        return frame[y:y+h, x:x+w]
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
            return cv2.resize(face, (96,96), interpolation=cv2.INTER_LINEAR)/255.
        c = raw_input("No face was detected, want to try again?([y]/n) : ").strip().lower()
    return None
