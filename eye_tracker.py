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
        for (x, y, w, h) in faces:
            frame = img[y:y + h, x:x + w]  # Remove the background
            # Draw Rectangle around Face
            img = cv2.rectangle(
                img, (x, y), (x + w, y+(h//2)), (255, 255, 0), 3)
            return frame
    else:
        return img


def left_eye_detection(img):
    eyes = LEFT_EYE_CASCADE.detectMultiScale(img, 1.05, 5)
    if eyes is not None:
        for (ex, ey, ew, eh) in eyes:
            ey = adjust_eye_position(ey, eh)
            eh = adjust_eye_height(eh)
            frame = img[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 255), 1)
            return frame


def right_eye_detection(img):
    eyes = RIGHT_EYE_CASCADE.detectMultiScale(img, 1.05, 5)
    if eyes is not None:
        print(eyes)
        for (ex, ey, ew, eh) in eyes:
            ey = adjust_eye_position(ey, eh)
            eh = adjust_eye_height(eh)
            frame = img[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (155, 0, 155), 1)
            return frame

def adjust_eye_height(eh):
    return int(round(eh // 2))
    

def adjust_eye_position(ey, eh):
    return int(round(ey + (eh / 2.5)))

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints

def nothing(x):
    pass


def main():

    cap = cv2.VideoCapture(0)
    # cv2.namedWindow('Eye Mouse', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Eye Mouse')
    cv2.createTrackbar('threshold', 'Eye Mouse', 36, 255, nothing)
    # cv2.createTrackbar('Frame Rate', 'Eye Mouse', 1, 10, nothing)
    img = cv2.imread('picture.jpg')

    while True:
        # cv2.waitKey(cv2.getTrackbarPos('Frame Rate', 'Eye Mouse') * 100)
        threshold = r = cv2.getTrackbarPos('threshold', 'image')
        # _, frame = cap.read()
        face_frame = face_detection(img)
        print(face_frame)
        if face_frame is not None:
            
            right_eye_frame = right_eye_detection(face_frame)
            if right_eye_frame is not None:
                print('found right eye')
                keypoints = blob_process(right_eye_frame, threshold, DETECTOR)
                right_eye_frame = cv2.drawKeypoints(right_eye_frame, keypoints, right_eye_frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('Eye Mouse', right_eye_frame)

            left_eye_frame = left_eye_detection(face_frame)
            if left_eye_frame is not None:
                print('found left eye')
        # cv2.imshow('Eye_Mouse.png', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
