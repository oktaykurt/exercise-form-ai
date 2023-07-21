import datetime
import IPython
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

for i in range(1, 11):
    # Input video file path
    video_file_path = f"captured-videos/squat{i}.mp4"

    # Initiate MediaPipe Pose process
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    model_path = "models/pose_landmarker_heavy.task"

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)

    # Initiate MediaPipe Drawing utility
    mp_draw = mp.solutions.drawing_utils 

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

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
            
            # print(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
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

            frame_time = datetime.datetime.now().strftime("%H:%M:%S.%f")

            pose_coordinates = [frame_time, left_hip, left_knee, left_ankle, left_shoulder, left_elbow, left_wrist, right_hip, right_knee, right_ankle, right_shoulder, right_elbow, right_wrist]
            pose_coordinates_list.append(pose_coordinates)
            
            # mp_draw.draw_landmarks(flipped_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_draw.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

            

            
        display_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('MediaPipe Pose', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    pose_coordinates_df = pd.DataFrame(pose_coordinates_list)
    pose_coordinates_df.columns = ["frame_time", "left_hip", "left_knee", "left_ankle", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "right_shoulder", "right_elbow", "right_wrist"]
    pose_coordinates_df.set_index("frame_time", inplace=True)

    # Display the DataFrame
    IPython.display.display(pose_coordinates_df)
    print(pose_coordinates_df)

    # pose_coordinates_df.to_csv(f"pose-coordinates/squat_df{i}.csv")