import datetime
import IPython
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Input video file path
video_file_path = "outpy01.mp4"


# Initiate MediaPipe Pose process
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initiate MediaPipe Drawing utility
mp_draw = mp.solutions.drawing_utils 

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle 

def draw_angle(image, point1, point2, point3, good_angles, bad_form, feedback_y_position, color_good=(0, 255, 0), color_bad=(0, 0, 255), thickness=2):
    angle = calculate_angle(point1, point2, point3)
    feedback = "Good form"
    color = color_good
    if angle < good_angles[0]:
        feedback = bad_form[0]
        color = color_bad
    elif angle > good_angles[1]:
        feedback = bad_form[1]
        color = color_bad

    cv2.putText(image, feedback, (10, feedback_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)

    # Draw the angle at the mid point (point2)
    cv2.putText(image, str(int(angle)), tuple(np.multiply(point2, [image.shape[1], image.shape[0]]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness, cv2.LINE_AA)

    # Draw lines connecting the points
    cv2.line(image, tuple(np.multiply(point1, [image.shape[1], image.shape[0]]).astype(int)), 
             tuple(np.multiply(point2, [image.shape[1], image.shape[0]]).astype(int)), color, thickness=8)
    
    cv2.line(image, tuple(np.multiply(point2, [image.shape[1], image.shape[0]]).astype(int)), 
             tuple(np.multiply(point3, [image.shape[1], image.shape[0]]).astype(int)), color, thickness=8)

    return image

# Open the video file
cap = cv2.VideoCapture(video_file_path)

feedback_y_position = 30

pose_coordinates_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB, and flip the image since it was originally mirrored
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    flipped_image = cv2.flip(rgb_image, 1)
    
    results = pose.process(cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR))

    if results.pose_landmarks:
        # get the landmarks
        landmarks = results.pose_landmarks.landmark
        
        # capture relevant points
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        current_time = datetime.datetime.now()

        pose_coordinates = ["squat", current_time, left_hip, left_knee, left_ankle, left_shoulder, left_elbow, left_wrist, right_hip, right_knee, right_ankle, right_shoulder, right_elbow, right_wrist]
        pose_coordinates_list.append(pose_coordinates)

        # Define the "good" angle ranges
        good_angles_hip_knee_ankle = [80, 140] 
        bad_form_hip_knee_ankle = ["Deep squat!", "Knee over toes!"]

        good_angles_shoulder_hip_knee = [100, 120] 
        bad_form_shoulder_hip_knee = ["Bending backwards!", "Bending forwards!"]

        good_angles_wrist_elbow_shoulder = [150, 200]
        bad_form_wrist_elbow_shoulder = ["Straight your arms!", "Straight your arms!"]

        # Draw angles at the joints with the defined "good" angle ranges
        draw_angle(flipped_image, left_shoulder, left_hip, left_knee, good_angles_shoulder_hip_knee, bad_form_shoulder_hip_knee, 100)
        draw_angle(flipped_image, left_hip, left_knee, left_ankle, good_angles_hip_knee_ankle, bad_form_hip_knee_ankle, 200)
        draw_angle(flipped_image, left_wrist, left_elbow, left_shoulder, good_angles_wrist_elbow_shoulder, bad_form_wrist_elbow_shoulder, 300)

    display_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('MediaPipe Pose', display_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()

pose_coordinates_df = pd.DataFrame(pose_coordinates_list)
pose_coordinates_df.set_index(0, inplace=True)

# Display the DataFrame
IPython.display.display(pose_coordinates_df)
print(pose_coordinates_df)
