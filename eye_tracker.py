import cv2
import numpy as np


# init part
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
LEFT_EYE_CASCADE = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
RIGHT_EYE_CASCADE = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
DETECTOR_PARAMS = cv2.SimpleBlobDetector_Params()
DETECTOR_PARAMS.filterByArea = True
DETECTOR = cv2.SimpleBlobDetector_create(DETECTOR_PARAMS)


def face_detection(img):
    # Make Face Grayscale
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray_picture, 1.05, 5)
    if faces is not None:
        # print "Face: " + faces
        for (x, y, w, h) in faces:
            frame = img[y:y + h, x:x + w]  # Remove the background
            # Draw Rectangle around Face
            img = cv2.rectangle(
                img, (x, y), (x + w, y+(h//2)), (255, 255, 0), 3)
        return frame


def left_eye_detection(img):
    eyes = LEFT_EYE_CASCADE.detectMultiScale(img, 1.05, 5)
    if eyes is not None:
        # print "Eyes: " + eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 255), 3)


def right_eye_detection(img):
    eyes = RIGHT_EYE_CASCADE.detectMultiScale(img, 1.05, 5)
    if eyes is not None:
        # print "Eyes: " + eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (155, 0, 155), 3)


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Eye Mouse', cv2.WINDOW_KEEPRATIO)
    # cv2.createTrackbar('threshold', 'Eye Mouse', 36, 255, nothing)
    cv2.createTrackbar('Frame Rate', 'Eye Mouse', 1, 10, nothing)
    while True:
        cv2.waitKey(cv2.getTrackbarPos('Frame Rate', 'Eye Mouse') * 100)
        _, frame = cap.read()
        face_frame = face_detection(frame)
        # if face_frame is not None:
        #     right_eye_frame = right_eye_detection(face_frame)
        #     if right_eye_frame is not None:
        #         print('found right eye')
            
        #     left_eye_frame = left_eye_detection(face_frame)
        #     if right_eye_frame is not None:
        #         print('found right eye')

        cv2.imshow('Eye Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
