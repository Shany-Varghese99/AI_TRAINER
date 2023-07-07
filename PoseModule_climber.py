import cv2
import mediapipe as mp
import time
import math
import numpy as np




# class PoseModule:
#     def _init_(self):
#         self.detector = self.poseDetector()
class poseDetector():

    def __init__(self, mode=False, modelComplexity=1, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        # def poseDetector(self,mode=False,modelComplexity=1, upBody=False, smooth=True,
        #              detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComplex = modelComplexity
        self.upBody = upBody
        self.smooth = smooth

        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplex, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        # Implement your pose detector initialization here
        pass

    def findPose(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame

        # Implement your pose detection code here

    # pass

    def findPosition(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    # Implement your pose position detection code here
    # pass

    def findAngle(self, frame, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
        # Implement your angle calculation code here
        # pass

    def processVideo(self, video_path):
        video = cv2.VideoCapture(1)

        # video = cv2.VideoCapture(video_path)
        count = 0
        dir = 0

        # detector = poseDetector()
        p_time = 0

        while True:
            success, frame = video.read()
            frame = cv2.flip(frame, 1)
            if not success:
                break

            # frame = cv2.flip(frame, 1)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # face_cascade = cv2.CascadeClassifier('/home/user/DL/INTERNSHIP/haarcascade_frontalface_default.xml')
            # face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

            frame = self.findPose(frame, draw=False)
            lmlist = self.findPosition(frame, draw=False)

            if len(lmlist) != 0:
                angle1 = self.findAngle(frame, 23, 25, 27)
                # print(angle1)

                low = 200
                high = 300

                per = np.interp(angle1, (low, high), (0, 100))

                bar = np.interp(angle1, (50, 160), (650, 100))

                color = (0, 255, 0)
                if per == 0:
                    color = (255, 0, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                if per == 100:
                    color = (255, 0, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1

                frame = cv2.rectangle(frame, (1100, 100), (1175, 650), color, 3)
                frame = cv2.rectangle(frame, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                frame = cv2.putText(frame, f'{int(per)} %', (1110, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

                frame = cv2.putText(frame, "COUNT", (30, 550), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.putText(frame, str(int(count)), (70, 590), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(frame, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow("Climbers Count", frame)

            # for (x, y, w, h) in face:
            #     cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=3)

            cv2.imshow("Climbers Count", frame)

            if cv2.waitKey(1) & 0XFF == 27:
                break

        video.release()
        cv2.destroyAllWindows()
