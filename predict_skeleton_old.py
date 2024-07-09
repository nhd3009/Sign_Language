import traceback
import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model

model = load_model('model/model_1.h5')
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

listA_Z = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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

                # if ch1 == 18 or ch1 == 0 or ch1 == 19 or ch1 == 4 or ch1 == 12 or ch1 == 13:
                #     ch1 = 'S'
                #     if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                #         ch1 = 'A'
                #     if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1] :
                #         ch1 = 'T'
                #     if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                #         ch1 = 'E'
                #     if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and  pts[4][1] < pts[18][1]:
                #         ch1 = 'M'
                #     if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0]  and  pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                #         ch1 = 'N'
                #
                # if ch1 == 14 or ch1 == 2:
                #     if distance(pts[12], pts[4]) > 42:
                #         ch1 = 'C'
                #     else:
                #         ch1 = 'O'
                #
                # if ch1 == 6 or ch1 == 7:
                #     if (distance(pts[8], pts[12])) > 72:
                #         ch1 = 'G'
                #     else:
                #         ch1 = 'H'
                #
                # if ch1 == 24 or ch1 == 9:
                #     if distance(pts[8], pts[4]) > 42:
                #         ch1 = 'Y'
                #     else:
                #         ch1 = 'J'
                #
                # if ch1 == 11:
                #     ch1 = 'L'
                #
                # if ch1 == 23:
                #     ch1 = 'X'
                #
                # if ch1 == 25 or ch1 == 16 or ch1 == 15:
                #     if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                #         if pts[8][1] < pts[5][1]:
                #             ch1 = 'Z'
                #         else:
                #             ch1 = 'Q'
                #     else:
                #         ch1 = 'P'
                #
                # if ch1 == 1 or ch1 == 3 or ch1 == 5 or ch1 == 8 or ch1 == 22 or ch1 == 10 or ch1 == 20 or ch1 == 21 or ch1 == 17:
                #     if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] >pts[20][1]):
                #         ch1 = 'B'
                #     if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                #         ch1 = 'D'
                #     if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                #         ch1 = 'F'
                #     if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                #         ch1 = 'I'
                #     if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
                #         ch1 = 'W'
                #     if  (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]:
                #         ch1 = 'K'
                #     if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                #         ch1 = 'U'
                #     if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] >pts[9][1]):
                #         ch1 = 'V'
                #
                #     if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                #         ch1 = 'R'

                # if ch1 != 1:
                #     if (ch1,ch2) in dicttt:
                #         dicttt[(ch1,ch2)] += 1
                #     else:
                #         dicttt[(ch1,ch2)] = 1


                frame = cv2.putText(frame, "Predicted " + listA_Z[ch1], (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    3, (0, 0, 255), 2, cv2.LINE_AA)


        cv2.imshow("Image", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


    except Exception:
        print("==", traceback.format_exc())


# dicttt = {key: val for key, val in sorted(dicttt.items(), key = lambda ele: ele[1], reverse = True)}
# print(dicttt)
# print(set(kok))
capture.release()
cv2.destroyAllWindows()