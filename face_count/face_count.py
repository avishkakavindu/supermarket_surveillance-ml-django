import cv2
import numpy as np
import dlib


def detect_faces(frame):
    detector = dlib.get_frontal_face_detector()
    # gray scaling
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = detector(gray)

    return len(faces), faces


def draw_rectangles(frame, faces):
    for i, face in enumerate(faces):
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Increment the iterartor each time you get the coordinates
        i = i + 1

        # Adding face number to the box detecting faces
        cv2.putText(frame, 'face num' + str(i), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # print(face, i)
    return frame


frame = cv2.imread('imgs/faces.png')

num_of_faces, faces = detect_faces(frame)

frame = draw_rectangles(frame, faces)

# Display the resulting frame
cv2.imshow('frame', frame)
cv2.waitKey(0)
