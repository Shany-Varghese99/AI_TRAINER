import cv2
from PoseModule_climber import poseDetector


video_path = cv2.VideoCapture(0)
pose_detector = poseDetector()
pose_detector.processVideo(video_path)




