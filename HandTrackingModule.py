import time
import numpy as np
import os

import cv2
import mediapipe as mp


gesture_commands = {
    "peace": lambda : os.system("xdotool key space"),
    "left_thumb": lambda : os.system("xdotool key Tab")
}

hand_gestures = {
    "left_thumb": [[0, 206, 330], [1, 241, 312], [2, 274, 282], [3, 308, 262], [4, 333, 254], [5, 255, 241], [6, 253, 217], [7, 253, 248], [8, 252, 255], [9, 228, 245], [10, 228, 229], [11, 232, 269], [12, 234, 264], [13, 203, 251], [14, 205, 238], [15, 212, 272], [16, 214, 271], [17, 181, 259], [18, 182, 249], [19, 191, 271], [20, 193, 275]],
    "peace" : [[0, 181, 383], [1, 224, 378], [2, 243, 339], [3, 219, 299], [4, 191, 266], [5, 255, 275], [6, 283, 216], [7, 299, 181], [8, 312, 151], [9, 216, 266], [10, 227, 198], [11, 235, 157], [12, 241, 121], [13, 180, 276], [14, 169, 230], [15, 179, 268], [16, 188, 297], [17, 149, 301], [18, 146, 279], [19, 163, 310], [20, 177, 335]]
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


def do_hand_control(current_hand_vector, tolerance=0.2):
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
        #cv2.imshow("Image", img)

        # Check for controls
        k = cv2.waitKey(1) & 0xFF

        # Escape to exit the loop
        if k == 27:
            break

        # Space to print out the current hand vector
        if k == 32:
            print(lmList)

        # try and detect a gesture
        do_hand_control(lmList)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
