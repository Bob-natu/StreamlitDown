import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile

# MediaPipeの設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Streamlitで動画をアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 一時的に動画ファイルを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        output_video_path = temp_file.name
        with open(output_video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

    # 動画の読み込み
    cap = cv2.VideoCapture(output_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画の準備
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out_file:
        output_processed_video_path = temp_out_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_processed_video_path, fourcc, 30.0, (frame_width, frame_height))

    # 動画処理と骨格抽出
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipeで骨格を処理
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 骨格が検出された場合、ランドマークを描画
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 処理後のフレームを動画ファイルに書き込み
        out.write(frame)

    cap.release()
    out.release()

    # 処理後の動画をStreamlitで表示
    with open(output_processed_video_path, 'rb') as f:
        video_bytes = f.read()
    st.video(video_bytes)
