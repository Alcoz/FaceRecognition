
import sys
import dlib
import cv2
import numpy as np
from PIL import Image
from imutils import face_utils

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "../utils/shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
file_name = "test4.jpg"
file_path = "../data/img/"
FILE = file_path + file_name
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

# Load the image
img = cv2.imread(FILE)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Run the HOG face detector on the image data
detected_faces = face_detector(gray, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), FILE))

array = []
# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates 
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
    shape = face_pose_predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)
    
    for (x, y) in shape:
        cv2.circle(gray, (x, y), 2, (255, 255, 255), -1)

    array = np.array(gray)

img = Image.fromarray(array, 'L')
img.save("../data/landmarks/" + file_name)
img.show()
