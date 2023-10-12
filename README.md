Exercise Form AI

This project utilizes Google's MediaPipe model to perform pose estimation on videos and classify exercise movements. The pose coordinates for various exercise movements are stored in CSV format, which are then used to train a classification model.

Repository Structure

capture_video.py: Captures video feed from the camera.
combine-csv.py: Utility script to combine multiple CSVs into a single file.
combined_exercise_data.csv: Aggregated dataset of pose coordinates for various exercises.
exercise_classifier.ipynb: Jupyter notebook that contains the code for training the exercise classification model.
plot.ipynb: Jupyter notebook for visualizing pose estimation results.
pose_estimation_from_video.py: Extracts pose coordinates from a video file using MediaPipe.
real_time.py: Performs real-time pose estimation and exercise classification from a live video feed.
models/: Directory containing the pre-trained MediaPipe model.
pose-coordinates/: Directory containing CSV files for individual exercise movements.
Getting Started

Ensure you have all the necessary dependencies installed.
Clone the repository to your local machine.
Use capture_video.py to capture video or use pre-recorded videos.
Use pose_estimation_from_video.py to extract pose coordinates from the video.
Run the exercise_classifier.ipynb notebook to train the exercise classification model.
Once the model is trained, you can use real_time.py to perform real-time exercise classification.
Acknowledgments

This project utilizes Google's MediaPipe for pose estimation.
