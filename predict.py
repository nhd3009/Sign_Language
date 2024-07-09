import traceback
import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model

model = load_model('model/model_2.h5')
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)


offset = 15
step = 1
flag = False
imgWhite = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("white.jpg", imgWhite)


def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def distance_3d(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2) + ((x[2] - y[2]) ** 2))

bfh = 0
dicttt=dict()
count=0
kok=[]

while True:
    try:

        _, frame = capture.read()
        hands, frame = hd.findHands(frame, draw=False, flipType=True)
        white = cv2.imread("white.jpg")

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            hand2, im2 = hd2.findHands(imgCrop, draw=True, flipType=True)

            if hand2:
                hand = hand2[0]
                pts = hand['lmList']
                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                cv2.imshow("2", white)

                white = white.reshape(1, 400, 400, 3)
                prob = np.array(model.predict(white)[0], dtype='float32')
                ch1 = np.argmax(prob, axis=0)
                prob[ch1] = 0
                ch2 = np.argmax(prob, axis=0)
                prob[ch2] = 0


                if ch1 == 0:
                    ch1 = 'S'
                    if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                        ch1 = 'A'
                    if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1] :
                        ch1 = 'T'
                    if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                        ch1 = 'E'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and  pts[4][1] < pts[18][1]:
                        ch1 = 'M'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0]  and  pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                        ch1 = 'N'


                if ch1 == 2:
                    if distance(pts[12], pts[4]) > 42:
                        ch1 = 'C'
                    else:
                        ch1 = 'O'

                if ch1 == 3:
                    if (distance(pts[8], pts[12])) > 72:
                        ch1 = 'G'
                    else:
                        ch1 = 'H'

                if ch1 == 7:
                    if distance(pts[8], pts[4]) > 42:
                        ch1 = 'Y'
                    else:
                        ch1 = 'J'

                if ch1 == 4:
                    ch1 = 'L'

                if ch1 == 6:
                    ch1 = 'X'

                if ch1 == 5:
                    if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                        if pts[8][1] < pts[5][1]:
                            ch1 = 'Z'
                        else:
                            ch1 = 'Q'
                    else:
                        ch1 = 'P'

                if ch1 == 1:
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] >pts[20][1]):
                        ch1 = 'B'
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                        ch1 = 'D'
                    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'F'
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'I'
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'W'
                    if  (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]:
                        ch1 = 'K'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'U'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] >pts[9][1]):
                        ch1 = 'V'

                    if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'R'

                if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] <
                            pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = " "

                if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
                    if (pts[4][0] < pts[5][0]) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] >
                            pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = "next"

                if ch1 == 'Next' or True:
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] >
                        pts[16][0] and pts[0][0] > pts[20][0]) and (
                            pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] <
                            pts[16][1] and pts[4][1] < pts[20][1]) and (
                            pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] <
                            pts[14][1] and pts[4][1] < pts[18][1]):
                        ch1 = 'Backspace'




                frame = cv2.putText(frame, "Predicted " + str(ch1), (32, 82),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)


                frame = cv2.putText(frame, "Predicted " + str(ch1), (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow("Image", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


    except Exception:
        print("==", traceback.format_exc())


capture.release()
cv2.destroyAllWindows()