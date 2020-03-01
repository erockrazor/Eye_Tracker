import cv2
import numpy as np
import pyautogui

# init part
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
LEFT_EYE_CASCADE = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
RIGHT_EYE_CASCADE = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
DETECTOR_PARAMS = cv2.SimpleBlobDetector_Params()
DETECTOR_PARAMS.filterByArea = True
DETECTOR = cv2.SimpleBlobDetector_create(DETECTOR_PARAMS)


def face_detection(img):
    faces = FACE_CASCADE.detectMultiScale(img, 1.05, 5)
    if faces is not None:
        for (x, y, w, h) in faces:
            frame = img[y:y + h, x:x + w]  # Remove the background
            # Draw Rectangle around Face
            img = cv2.rectangle(
                img, (x, y), (x + w, y+(h//2)), (255, 255, 0), 3)
            return frame
    else:
        return img


def left_eye_detection(img, eye_height, eye_position):
    eyes = LEFT_EYE_CASCADE.detectMultiScale(img, 1.05, 5)
    if eyes is not None:
        for (ex, ey, ew, eh) in eyes:
            ey = adjust_eye_position(ey, eh, eye_position)
            eh = adjust_eye_height(eh, eye_height)
            frame = img[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 2)
            return frame


def right_eye_detection(img, eye_height, eye_position):
    eyes = RIGHT_EYE_CASCADE.detectMultiScale(img, 1.05, 5)
    if eyes is not None:
        for (ex, ey, ew, eh) in eyes:
            ey = adjust_eye_position(ey, eh, eye_position)
            eh = adjust_eye_height(eh, eye_height)
            frame = img[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 2)
            return frame


def pupil_detection(img, threshold):
    # _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    # Blur using 3 * 3 kernel. 
    # img = cv2.blur(img, (3, 3)) 
  
    # Apply Hough transform on the blurred image. 
    # detected_circles = cv2.HoughCircles(img,  cv2.HOUGH_GRADIENT, 1, 20) 
    # print(detected_circles)
    return img


def adjust_eye_height(eh, eye_height):
    return int(round(eh // eye_height))


def adjust_eye_position(ey, eh, eye_position):
    return int(round(ey + (eh / eye_position)))


def nothing(x):
    pass


def main():

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Eye Mouse', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('EyeMouse', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('EyeMouse', 400,400)
    cv2.createTrackbar('threshold', 'Eye Mouse', 90, 255, nothing)
    # cv2.createTrackbar('Frame Rate', 'Eye Mouse', 1, 10, nothing)
    cv2.createTrackbar('Eye Height', 'Eye Mouse', 22, 100, nothing)
    cv2.createTrackbar('Eye Position', 'Eye Mouse', 25, 100, nothing)

    while True:
        # cv2.waitKey(cv2.getTrackbarPos('Frame Rate', 'Eye Mouse') * 100)
        threshold = cv2.getTrackbarPos('threshold', 'Eye Mouse')
        eye_height = cv2.getTrackbarPos('Eye Height', 'Eye Mouse') * .1
        eye_position = cv2.getTrackbarPos('Eye Position', 'Eye Mouse') * .1
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_frame = face_detection(frame)
        if face_frame is not None:
            right_eye_frame = right_eye_detection(face_frame, eye_height, eye_position)
            left_eye_frame = left_eye_detection(face_frame, eye_height, eye_position)
            # if right_eye_frame is not None:
            #     img = pupil_detection(right_eye_frame, threshold)
            #     if img is not None:
            #         # cv2.drawKeypoints(right_eye_frame, keypoints, img, (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #         cv2.imshow('EyeMouse', img)
            if left_eye_frame is not None:
                img = pupil_detection(left_eye_frame, threshold)
                if img is not None:
                    # cv2.drawKeypoints(left_eye_frame, keypoints, img, (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow('EyeMouse', img)
        # pyautogui.moveTo(200, 200)
        
        # left_eye_frame = left_eye_detection(face_frame)
        # if left_eye_frame is not None:
        #     pupil = pupil_detection(left_eye_frame, threshold)
        #     if pupil is not None:
        #         print('pupil found')
        #     cv2.imshow('EyeMouse', left_eye_frame)
        # frame = cv2.GaussianBlur(frame, (7, 7), 0)
        # print(threshold)
        # _, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)
        # _, contours = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('Eye_Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
