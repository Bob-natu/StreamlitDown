import streamlit as st
import cv2 as cv
import mediapipe as mp
import tempfile
import os

def process_video(input_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    video = cv.VideoCapture(input_path)
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)
    
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        out.write(frame)
    
    video.release()
    out.release()
    pose.close()

def main():
    st.title("Pose Estimation Video Processing")
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_file.read())
            input_path = temp_input.name
        
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        st.info("Processing video... Please wait.")
        process_video(input_path, output_path)
        
        st.success("Processing complete!")
        
        with open(output_path, "rb") as file:
            st.download_button(label="Download Processed Video", data=file, file_name="processed_video.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
