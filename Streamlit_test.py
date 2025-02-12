import streamlit as st
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import ffmpeg
from io import BytesIO

def get_video_rotation(video_path):
    try:
        meta_data = ffmpeg.probe(video_path)
        for stream in meta_data['streams']:
            if stream['codec_type'] == 'video' and 'side_data_list' in stream:
                for side_data in stream['side_data_list']:
                    if side_data['side_data_type'] == 'Display Matrix':
                        rotation = side_data.get('rotation', 0)
                        return rotation
    except Exception as e:
        st.error(f"動画の回転情報を取得できませんでした: {e}")
    return 0

def correct_video_rotation(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

st.title("肩の位置追跡とグラフ作成")

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    input_video_path = os.path.join("/tmp", uploaded_file.name)
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    rotation = get_video_rotation(input_video_path)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けません: {input_video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    frame_numbers = []
    right_shoulder_y = []
    left_shoulder_y = []

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = correct_video_rotation(frame, rotation)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                
                frame_numbers.append(len(frame_numbers) + 1)
                right_shoulder_y.append(right_shoulder.y)
                left_shoulder_y.append(left_shoulder.y)
                
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            out.write(frame)
        
        cap.release()
        out.release()
    
    with open(output_video_path, "rb") as f:
        st.download_button("修正済み動画をダウンロード", f, file_name="corrected_video.mp4", mime="video/mp4")
