python
import cv2

# load video
cap = cv2.VideoCapture('path_to_video.mp4')

# initialize background subtraction model
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # read frame from video
    ret, frame = cap.read()

    # apply background subtraction
    fgmask = fgbg.apply(frame)

    # apply morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # find contours of moving objects
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show video
    cv2.imshow('frame', frame)

    # exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
