from facenet_pytorch import MTCNN
from PIL import Image
from skimage import io
from imutils import face_utils
import dlib
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
import sys
from openface import openface

def detect_faces(img, NAME):

    mtcnn = MTCNN(image_size=256, margin=0, keep_all=True)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path="saves/" + NAME)

    try:
        nb_face = len(img_cropped)
    except TypeError:
        print("No face detected")
        sys.exit(1)

    print(str(nb_face) + " faces detected")

    facelist = []
    facelist.append("saves/" + NAME)
    for i in range(nb_face-1):
       facelist.append("saves/" + NAME[:-4] + '_' + str(i+2) + ".jpg")

    return facelist

def process_landmarks(face):
    img = cv2.imread(face)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    # Detecting faces in the grayscale image
    rects = detector(gray, 1)
    # print(faces)
    predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    landmarks = None
    for rect in rects:
        # We will determine the facial landmarks for the face region, then
        # can convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks = shape

        # We then loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(gray, (x, y), 1, (255, 255, 255), -1)

        alignedFace = face_aligner.align(256, gray, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    
        print(np.array(alignedFace).shape)
        alignedFace = np.array(alignedFace)

        img = Image.fromarray(alignedFace, 'L')
        img.save(face)
        img.show()
        input("Press a key to continue...")
    # array = np.array(gray)
    # img = Image.fromarray(array, 'L')
    
    return landmarks


PATH = "img/"
NAME = "img3.jpg"
img = Image.open(PATH+NAME)

facelist = detect_faces(img, NAME)
marks = []
for face in facelist:
    marks.append(process_landmarks(face))

# print(marks)

    
