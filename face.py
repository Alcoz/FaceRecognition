import os
import sys
import dlib
import cv2
from openface import openface

# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "utils/shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

# Take the image file name from the command line
image_directory = "data/img/"

for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        FILE = image_directory + filename

        # Load the image
        image = cv2.imread(FILE)

        # Run the HOG face detector on the image data
        detected_faces = face_detector(image, 1)

        print("Found {} faces in the image file {}".format(len(detected_faces), FILE))

        # Loop through each face we found in the image
        for i, face_rect in enumerate(detected_faces):

            # Detected faces are returned as an object with the coordinates
            # of the top, left, right and bottom edges
            print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

            # Get the the face's pose
            pose_landmarks = face_pose_predictor(image, face_rect)

            # Use openface to calculate and perform the face alignment
            alignedFace = face_aligner.align(224, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            # Save the aligned image to a file
            if len(detected_faces) == 1:
                cv2.imwrite("data/aligned_faces/" +  filename, alignedFace)
            else:
                cv2.imwrite("data/aligned_faces/" +  filename[:-4] + "_{}.jpg".format(i), alignedFace)

# Arriver à lancer les models d'openface pour générer les embeddings