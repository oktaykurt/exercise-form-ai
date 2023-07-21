from collections import deque
import cv2
import mediapipe as mp
import torch
import torch.nn as nn


# Global variables
sequence = []
predictions = []
threshold = 0.85  # minimum confidence to classify an action/exercise
current_action = None
squat_counter = 0
lunge_counter = 0
push_up_counter = 0
squat_stage = None
lunge_stage = None
pushup_stage = None
current_exercise = None
label_map = {0: 'lunge', 1:'push-up',  2: 'squat'}  # Modify this to match your classes
cap = cv2.VideoCapture(0)  # Change this to 0 if you only have one camera

# Model parameters
n_features = 24
n_time_steps = 75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)  # Add this line
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batchnorm(out[:, -1, :])  # Add this line
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Load the saved parameters into the model
model = LSTMModel(n_features, 32, 3, 2, 0.2).to(device)  # Change the output_dim to the number of actions
model.load_state_dict(torch.load('exercise_classification_model.pth'))
model.eval()

# MediaPipe pose process
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize deque with maximum length of 75
window = deque(maxlen=75)

# Define the four functions
def mediapipe_detection(frame, model):
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = model.process(image)

    # Convert the RGB image to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_landmarks(image, results):
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

def extract_coordinates(results):
    if results.pose_landmarks is None:
        return []
    landmarks = results.pose_landmarks.landmark
    return [
        [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y],
    ]

def count_reps(image, current_action, landmarks, mp_pose):
    global squat_counter, squat_stage, lunge_counter, lunge_stage, push_up_counter, pushup_stage, current_exercise
    
    if current_action == current_exercise:
        # Calculate the average y coordinate of the hips
        avg_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y) / 2
        avg_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2

        # If the current action is a squat
        if current_action == 'squat':
            if (squat_stage is None or squat_stage == 'up') and avg_hip > 0.6:
                squat_stage = 'down'
            if squat_stage == 'down' and avg_hip < 0.5:
                squat_stage = 'up'
                squat_counter += 1

        # If the current action is a lunge
        elif current_action == 'lunge':
            if (lunge_stage is None or lunge_stage == 'up') and avg_hip > 0.6:
                lunge_stage = 'down'
            if lunge_stage == 'down' and avg_hip < 0.5:
                lunge_stage = 'up'
                lunge_counter += 1

        # If the current action is a lunge
        elif current_action == 'push-up':
            if (pushup_stage is None or pushup_stage == 'up') and avg_hip > 0.5:
                pushup_stage = 'down'
            if pushup_stage == 'down' and avg_hip < 0.4:
                pushup_stage = 'up'
                push_up_counter += 1

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Process the frame
    image, results = mediapipe_detection(image, pose)

    # Draw the landmarks
    if results.pose_landmarks:
        draw_landmarks(image, results)

        # Extract coordinates
        coordinates = extract_coordinates(results)
        if coordinates:
            window.append(coordinates)

            # If we have enough frames, make a prediction
            if len(window) == n_time_steps:
                sequence = list(window)
                sequence_tensor = torch.tensor(sequence).view(1, n_time_steps, n_features).to(device)
                with torch.no_grad():
                    prediction = model(sequence_tensor)
                    predicted_action = torch.argmax(prediction, dim=1).item()
                    confidence = (prediction[0, predicted_action].item())
                    
                    # Check if the confidence of the prediction is greater than the threshold

                    print(prediction)

                    if confidence > threshold:
                        current_action = label_map[predicted_action]
                    else:
                        current_action = None  # No current exercise if confidence is below the threshold

                    # Update the current exercise
                    if current_action != current_exercise:
                        current_exercise = current_action

                    # Count reps
                    count_reps(image, current_action, results.pose_landmarks.landmark, mp_pose)
                    # Display the number of reps
                    cv2.putText(image, 'Squat: ' + str(squat_counter), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
                    cv2.putText(image, 'Lunge: ' + str(lunge_counter), (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
                    cv2.putText(image, 'Push-up: ' + str(push_up_counter), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)

                    # cv2.putText(image, 'Prediction: {}'.format(current_action), (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
                    # cv2.putText(image, 'Confidence: {}'.format(confidence), (30, 275), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)

    # Display the image
    cv2.imshow('Exercise Counter', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
