import cv2
import mediapipe as mp
import streamlit as st
import tempfile

# MediaPipeの設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 動画の読み込みと処理
video_path = 'input_video.mp4'  # ここに実際の動画パスを指定

# 一時的な動画ファイルを作成
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
    output_video_path = temp_file.name

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の準備
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipeで骨格を処理
    results = pose.process(image)

    # 骨格が検出された場合
    if results.pose_landmarks:
        # 骨格を描画
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 動画にフレームを書き込み
    out.write(frame)

cap.release()
out.release()

# 動画ファイルを一時的に保存
with open(output_video_path, 'rb') as f:
    video_bytes = f.read()

# Streamlitで動画を表示
st.video(video_bytes)
