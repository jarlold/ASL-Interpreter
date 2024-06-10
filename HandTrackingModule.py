import time
import numpy as np
import os

import cv2
import mediapipe as mp


gesture_commands = {
    "up_thumb": lambda : os.system("xdotool key Return"),
    "left_thumb": lambda : os.system("xdotool key Tab"),
    "down_thumb": lambda : os.system("xdotool click 1")
}

hand_gestures = {
    "left_thumb": [[0, 206, 330], [1, 241, 312], [2, 274, 282], [3, 308, 262], [4, 333, 254], [5, 255, 241], [6, 253, 217], [7, 253, 248], [8, 252, 255], [9, 228, 245], [10, 228, 229], [11, 232, 269], [12, 234, 264], [13, 203, 251], [14, 205, 238], [15, 212, 272], [16, 214, 271], [17, 181, 259], [18, 182, 249], [19, 191, 271], [20, 193, 275]],
    "up_thumb" : [[-10.0, -97.14285714285714, 48.19047619047618], [-9.0, -76.14285714285714, -4.809523809523818], [-8.0, -38.142857142857146, -62.80952380952382], [-7.0, -23.142857142857142, -108.80952380952382], [-6.0, -23.142857142857142, -147.8095238095238], [-5.0, -11.142857142857142, -48.80952380952382], [-4.0, 61.857142857142854, -30.809523809523817], [-3.0, 41.857142857142854, -21.809523809523817], [-2.0, 15.857142857142858, -24.809523809523817], [-1.0, -11.142857142857142, -10.809523809523819], [0.0, 54.857142857142854, 5.190476190476182], [1.0, 31.857142857142858, 8.190476190476181], [2.0, 6.857142857142857, 3.190476190476182], [3.0, -13.142857142857142, 29.190476190476183], [4.0, 44.857142857142854, 37.19047619047618], [5.0, 22.857142857142858, 38.19047619047618], [6.0, -1.1428571428571428, 35.19047619047618], [7.0, -15.142857142857142, 66.19047619047618], [8.0, 29.857142857142858, 63.19047619047618], [9.0, 9.857142857142858, 64.19047619047618], [10.0, -11.142857142857142, 63.19047619047618]],
    "right_thumb": [[-10.0, 24.428571428571416, 153.33333333333334], [-9.0, -51.57142857142858, 97.33333333333333], [-8.0, -101.57142857142858, 28.333333333333336], [-7.0, -142.57142857142858, -32.666666666666664], [-6.0, -179.57142857142858, -73.66666666666667], [-5.0, -35.57142857142858, -65.66666666666667], [-4.0, -41.57142857142858, -48.666666666666664], [-3.0, -37.57142857142858, -12.666666666666668], [-2.0, -36.57142857142858, 10.333333333333332], [-1.0, 22.428571428571416, -59.666666666666664], [0.0, 6.428571428571419, -39.666666666666664], [1.0, 2.428571428571419, -9.666666666666668], [2.0, 3.428571428571419, 15.333333333333332], [3.0, 70.42857142857142, -35.666666666666664], [4.0, 48.42857142857142, -18.666666666666664], [5.0, 43.42857142857142, 7.333333333333333], [6.0, 42.42857142857142, 32.333333333333336], [7.0, 108.42857142857142, -5.666666666666667], [8.0, 91.42857142857142, -1.6666666666666672], [9.0, 82.42857142857142, 19.333333333333336], [10.0, 80.42857142857142, 40.333333333333336]],
    "down_thumb": [[-10.0, -76.04761904761905, -31.47619047619047], [-9.0, -65.04761904761905, 14.523809523809529], [-8.0, -38.04761904761905, 63.52380952380953], [-7.0, -18.047619047619047, 103.52380952380952], [-6.0, -8.04761904761905, 139.52380952380952], [-5.0, 46.95238095238095, 36.52380952380953], [-4.0, 20.952380952380953, 27.52380952380953], [-3.0, -10.04761904761905, 22.52380952380953], [-2.0, -5.047619047619048, 25.52380952380953], [-1.0, 56.95238095238095, 2.5238095238095286], [0.0, 16.952380952380953, -9.476190476190471], [1.0, -17.047619047619047, -4.476190476190472], [2.0, -6.047619047619048, -1.4761904761904716], [3.0, 54.95238095238095, -31.47619047619047], [4.0, 14.95238095238095, -41.47619047619047], [5.0, -15.04761904761905, -33.47619047619047], [6.0, -4.047619047619048, -28.47619047619047], [7.0, 46.95238095238095, -62.47619047619047], [8.0, 17.952380952380953, -69.47619047619048], [9.0, -8.04761904761905, -62.47619047619047], [10.0, -6.047619047619048, -59.47619047619047]]
}

# Dont fuck with these I use them to stop the program
# from spamming the same command every frame
_p_gesture = None
_tlo_gesture = 100

class HandDectector():
    def __init__(self, mode=False, maxHands=2, complexitiy= 1, detectionConf = 0.5, trackConf = 0.5 ):
        '''
        :param mode: bool
        :param maxHands: int
        :param complexitiy: int
        :param detectionConf: int
        :param trackConf: int
        '''
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexitiy
        self.dConf = detectionConf
        self.tConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.dConf, self.tConf)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        '''
        :param img:  cap.read()
        :param draw: bool
        :return: img
        '''

        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # drawing hand points + lines
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.mpDraw.DrawingSpec(color=(255, 0, 0)))

        return img

    def findPosition(self, img, mark=0, draw=True):
        '''
        :param img:  cap.read()
        :param HandNum: int
        :param draw: bool
        :return: a list with all the landmark positions
        '''
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[mark]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                # drawing circle over hand landmark
                if draw:
                    cv2.circle(img, (cx,cy), 11, (0,255,255), cv2.FILLED)

        return lmList


def recentre_hand_vector(v1):
    avgx = 0
    avgy = 0
    avgz = 0
    for x, y, z in v1:
        avgx += x
        avgy += y
        avgz += z
    avgx /= len(v1)
    avgy /= len(v1)
    avgz /= len(v1)
    for i in range(len(v1)):
        v1[i][0] -= avgx
        v1[i][1] -= avgy
        v1[i][2] -= avgz

    return v1


def gesture_distance_euclidean(v1, v2):
    # Vectors have to be the same length and not empty
    if len(v1) != len(v2):
        return None
    if len(v1) == 0:
        return None

    # Then recentre them so the hand's translation matters less
    v1 = recentre_hand_vector(v1)
    v2 = recentre_hand_vector(v2)

    # convert them to numpy vector
    v1 = np.array(v1)
    v2 = np.array(v2)


    # normalize them
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    sum = 0
    for i in range(len(v1)):
        for j in range(0, 2):
            sum += (v1[i][j] - v2[i][j])**2
    return sum ** 0.5


def find_closest_gesture(hand_vector):
    closest_name = None
    closest_score = None
    for i in hand_gestures:
        # Get the difference between the gestures
        cur_score = gesture_distance_euclidean(hand_vector, hand_gestures[i])

        # Sometimes that's impossible, in which case skip along
        if cur_score is None:
            continue

        # Then see if that's any closer than the closest gesture
        if closest_score is None or cur_score < closest_score:
            closest_score = cur_score
            closest_name = i

    # Return the most similar gesture's name and how close it was
    return closest_name, closest_score


def do_hand_control(current_hand_vector, tolerance=0.20):
    global _p_gesture, _tlo_gesture

    # Find the closest gesture
    name, score = find_closest_gesture(current_hand_vector)

    # If we couldn't compute name or score then dont do shit
    if name is None or score is None:
        return None

    # If it's better than the tolerance, hurray we found him
    if score <= tolerance:
        # If we detect a different gesture, or its been more than two seconds
        # trigger the gesture function
        if name != _p_gesture or time.time() - _tlo_gesture > 2:
            print(name)
            try:
                gesture_commands[name]()
            except KeyError as e:
                print("Gesture " + name + " is not bound!")
            _tlo_gesture = time.time()
        _p_gesture = name
        return name

    # otherwise just return None
    return None
    

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDectector()

    while True:
        # running webcam
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        # Useful for debugging but not much else
        cv2.imshow("Image", img)

        # try and detect a gesture
        do_hand_control(lmList)

        # Check for controls
        k = cv2.waitKey(1) & 0xFF

        # Escape to exit the loop
        if k == 27:
            break

        # Space to print out the current hand vector
        if k == 32:
            print(lmList)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
