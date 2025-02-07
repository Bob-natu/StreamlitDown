import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

def process_video(input_video_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    min_y_value = float('inf')
    min_y_frame = 0
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                left_heel_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * height
                right_heel_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * height
                avg_y = (left_heel_y + right_heel_y) / 2
                if avg_y < min_y_value:
                    min_y_value = avg_y
                    min_y_frame = frame_count
    
    start_frame = max(0, min_y_frame - 0)
    end_frame = min(total_frames - 1, min_y_frame + 40)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    temp_video_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    extracted_frames = []
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                for lm in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]:
                    x, y = int(results.pose_landmarks.landmark[lm].x * width), int(results.pose_landmarks.landmark[lm].y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            extracted_frames.append(frame.copy())
            out.write(frame)
    
    cap.release()
    out.release()
    return temp_video_path, extracted_frames

st.title("Pose Extraction and Video Processing")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    
    st.video(temp_file_path)
    
    if st.button("Process Video"):
        with st.spinner("Processing..."):
            output_video_path, extracted_frames = process_video(temp_file_path)
            
        st.success("Processing complete!")
        st.subheader("Extracted Frames")
        for img in extracted_frames:
            st.image(img, channels="BGR")
        
        with open(output_video_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")
