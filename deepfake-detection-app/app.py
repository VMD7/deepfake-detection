import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import cv2
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import dlib
import tempfile
import shutil
import os
from tensorflow.keras.applications.inception_v3 import preprocess_input


model_path = 'deepfake_detection_model.h5'
model = tf.keras.models.load_model(model_path)

IMG_SIZE = (299, 299) 
MOTION_THRESHOLD = 20  
FRAME_SKIP = 2       
no_of_frames = 10
MAX_FRAMES=no_of_frames

detector = dlib.get_frontal_face_detector()

def extract_faces_from_frame(frame, detector):
    """
    Detects faces in a frame and returns the resized faces.

    Parameters:
    - frame: The video frame to process.
    - detector: Dlib face detector.

    Returns:
    - resized_faces (list): List of resized faces detected in the frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)
    resized_faces = []

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size != 0:  
            resized_face = cv2.resize(crop_img, IMG_SIZE)
            resized_faces.append(resized_face)

    # Debug: Log the number of faces detected
    #print(f"Detected {len(resized_faces)} faces in current frame")
    return resized_faces

def process_frame(video_path, detector, frame_skip):
    """
    Processes frames to extract motion and face data concurrently.

    Parameters:
    - cap: OpenCV VideoCapture object.
    - detector: Dlib face detector.
    - frame_skip (int): Number of frames to skip for processing.

    Returns:
    - motion_frames (list): List of motion-based face images.
    - all_faces (list): List of all detected faces for fallback.
    """
    prev_frame = None
    frame_count = 0
    motion_frames = []
    all_faces = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to improve processing speed
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Debug: Log frame number being processed
        #print(f"Processing frame {frame_count}")

        # # Resize frame to reduce processing time (optional, adjust size as needed)
        # frame = cv2.resize(frame, (640, 360))

        # Extract faces from the current frame
        faces = extract_faces_from_frame(frame, detector)
        all_faces.extend(faces)  # Store all faces detected, including non-motion

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = gray_frame
            frame_count += 1
            continue

        # Calculate frame difference to detect motion
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        motion_score = np.sum(frame_diff)

        # Debug: Log the motion score
        #print(f"Motion score: {motion_score}")

        # Check if motion is above the defined threshold and add the face to motion frames
        if motion_score > MOTION_THRESHOLD and faces:
            motion_frames.extend(faces)

        prev_frame = gray_frame
        frame_count += 1

    cap.release()
    return motion_frames, all_faces

def select_well_distributed_frames(motion_frames, all_faces, no_of_frames):
    """
    Selects well-distributed frames from the detected motion and fallback faces.

    Parameters:
    - motion_frames (list): List of frames with detected motion.
    - all_faces (list): List of all detected faces.
    - no_of_frames (int): Required number of frames.

    Returns:
    - final_frames (list): List of selected frames.
    """
    # Case 1: Motion frames exceed the required number
    if len(motion_frames) >= no_of_frames:
        interval = len(motion_frames) // no_of_frames
        distributed_motion_frames = [motion_frames[i * interval] for i in range(no_of_frames)]
        return distributed_motion_frames

    # Case 2: Motion frames are less than the required number
    needed_frames = no_of_frames - len(motion_frames)

    # If all frames together are still less than needed, return all frames available
    if len(motion_frames) + len(all_faces) < no_of_frames:
        #print(f"Returning all available frames: {len(motion_frames) + len(all_faces)}")
        return motion_frames + all_faces

    interval = max(1, len(all_faces) // needed_frames)
    additional_faces = [all_faces[i * interval] for i in range(needed_frames)]

    combined_frames = motion_frames + additional_faces
    interval = max(1, len(combined_frames) // no_of_frames)
    final_frames = [combined_frames[i * interval] for i in range(no_of_frames)]
    return final_frames

def extract_frames(no_of_frames, video_path):
  motion_frames, all_faces = process_frame(video_path, detector, FRAME_SKIP)
  final_frames = select_well_distributed_frames(motion_frames, all_faces, no_of_frames)
  return final_frames


def predict_video(model, video_path):
    """
    Predict if a video is REAL or FAKE using the trained model.

    Parameters:
    - model: The loaded deepfake detection model.
    - video_path: Path to the video file to be processed.

    Returns:
    - str: 'REAL' or 'FAKE' based on the model's prediction.
    """
    # Extract frames from the video
    frames = extract_frames(no_of_frames, video_path)
    original_frames = frames

    # Convert the frames list to a 5D tensor (1, time_steps, height, width, channels)
    if len(frames) < MAX_FRAMES:
        # Pad with zero arrays to match MAX_FRAMES
        while len(frames) < MAX_FRAMES:
            frames.append(np.zeros((299, 299, 3), dtype=np.float32))

    frames = frames[:MAX_FRAMES] 
    frames = np.array(frames)    
    frames = preprocess_input(frames) 

    # Expand dims to fit the model input shape
    input_data = np.expand_dims(frames, axis=0)  # Shape becomes (1, MAX_FRAMES, 299, 299, 3)

    # Predict using the model
    prediction = model.predict(input_data)
    probability = prediction[0][0]  # Get the probability for the first (and only) sample
    # Convert probability to class label
    if probability >=0.6:
        predicted_label='FAKE'
    else:
        predicted_label = 'REAL'
        probability=1-probability
    return original_frames, predicted_label, probability

def display_frames_and_prediction(video_file):
    # Ensure file is completely uploaded and of correct type
    if video_file is None:
        return [], "<div style='color: red;'>No video uploaded!</div>", ""
    
    # Check file size (10 MB limit)
    if os.path.getsize(video_file) > 10 * 1024 * 1024:  # 10 MB
        return [], "<div style='color: red;'>File size exceeds 10 MB limit!</div>", ""
    
    # Check file extension
    if not video_file.endswith('.mp4'):
        return [], "<div style='color: red;'>Only .mp4 files are allowed!</div>", ""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file_path = temp_file.name

    with open(video_file, 'rb') as src_file:
        with open(temp_file_path, 'wb') as dest_file:
            shutil.copyfileobj(src_file, dest_file)

    frames, predicted_label, confidence = predict_video(model, temp_file_path)
    os.remove(temp_file_path)

    confidence_text = f"Confidence: {confidence:.2%}"

    prediction_style = (
    f"<div style='color: {'green' if predicted_label == 'REAL' else 'red'}; "
    "text-align: center; font-size: 24px; font-weight: bold; "
    "border: 2px solid; padding: 10px; border-radius: 5px;'>"
    f"{predicted_label}</div>"
)


    return frames, prediction_style, confidence_text

iface = gr.Interface(
    fn=display_frames_and_prediction,
    inputs=gr.File(label="Upload Video", interactive=True),
    outputs=[
        gr.Gallery(label="Extracted Frames"), 
        gr.HTML(label="Prediction"), 
        gr.Textbox(label="Confidence", interactive=False)
    ],
    title="Deepfake Detection",
    description="Upload a video to determine if it is REAL or FAKE based on the deepfake detection model.",
    css="app.css",
    examples=[
        ["examples/abarnvbtwb.mp4"], 
        ["examples/aapnvogymq.mp4"],
    ],
    cache_examples="lazy"
)

iface.launch()