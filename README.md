# Exercise Form AI

This project utilizes Google's MediaPipe model to perform pose estimation on videos and classify exercise movements. The pose coordinates for various exercise movements are stored in CSV format, which are then used to train a classification model.

<img width="829" alt="app_screenshot" src="https://github.com/oktaykurt/exercise-form-ai/assets/10723547/67a2e62d-f40a-40ff-bbe5-5e2da834f3c1">


## Repository Structure
- `capture_video.py`: Captures video feed from the camera.
- `combine-csv.py`: Utility script to combine multiple CSVs into a single file.
- `combined_exercise_data.csv`: Aggregated dataset of pose coordinates for various exercises.
- `exercise_classifier.ipynb`: Jupyter notebook that contains the code for training the exercise classification model.
- `plot.ipynb`: Jupyter notebook for visualizing pose estimation results.
- `pose_estimation_from_video.py`: Extracts pose coordinates from a video file using MediaPipe.
- `real_time.py`: Performs real-time pose estimation and exercise classification from a live video feed.
- `models/`: Directory containing the pre-trained MediaPipe model.
- `pose-coordinates/`: Directory containing CSV files for individual exercise movements.

## Getting Started

1. Ensure you have all the necessary dependencies installed.
2. Clone the repository to your local machine.
3. Use `capture_video.py` to capture video or use pre-recorded videos.
4. Use `pose_estimation_from_video.py` to extract pose coordinates from the video.
5. Run the `exercise_classifier.ipynb` notebook to train the exercise classification model.
6. Once the model is trained, you can use `real_time.py` to perform real-time exercise classification.

## Acknowledgments

This project utilizes [Google's MediaPipe](https://mediapipe.dev/) for pose estimation.
