import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import numpy as np

# Streamlit UI 設定
st.title("🏐 Volleyball Jump Analysis with MediaPipe")

# ファイルアップロード
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    # 一時ファイルに保存
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # 動画の基本情報を取得
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS は小数のまま保持
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.write(f"**動画情報:** {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # MediaPipe 設定
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    
    # 最も低い位置のフレームを探す
    min_y_value = float('inf')
    min_y_frame = 0
    
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
    
    cap.release()
    
    # 切り取り範囲を計算
    start_frame = max(0, min_y_frame - 0)
    end_frame = min(total_frames - 1, min_y_frame + 40)
    
    # 処理後の動画を保存する一時ファイル
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    cap = cv2.VideoCapture(temp_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 骨格解析して新しい動画を作成
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            for landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                             mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL]:
                lm = results.pose_landmarks.landmark[landmark]
                x, y = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    st.success("✅ 骨格解析が完了しました！")
    
    # 処理後の動画を表示
    st.video(output_video_path)
    
    # ダウンロードボタン
    with open(output_video_path, "rb") as f:
        st.download_button(label="📥 処理済み動画をダウンロード", data=f, file_name="processed_video.mp4", mime="video/mp4")
