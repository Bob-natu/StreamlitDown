import cv2
import mediapipe as mp
import os
import tempfile
import numpy as np
import streamlit as st

# Streamlit UI
st.title("Volleyball Spike Analysis App")
uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "mov", "avi"])

# MediaPipe Poseの初期化
mp_pose = mp.solutions.pose
pose_connections = mp_pose.POSE_CONNECTIONS

if uploaded_file is not None:
    # 一時ファイルとして保存
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    # 動画を読み込む
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレート（小数点対応）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 幅
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高さ
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数

    st.write(f"📊 動画情報: FPS={fps:.2f}, 解像度={width}x{height}, フレーム数={total_frames}")

    # === 最小Y座標を探す（最も低い位置のフレームを特定） ===
    min_y_value = float('inf')
    min_y_frame = 0
    landmark_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL
    ]

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
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

    # 切り取りの開始・終了フレームを計算
    start_frame = max(0, min_y_frame - 0)
    end_frame = min(total_frames - 1, min_y_frame + 40)

    # 出力動画の設定（保存用）
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 切り取ったフレームのリスト
    extracted_frames = []

    # 切り取った範囲の動画を書き込み（骨格描画あり）
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 骨格描画
            if results.pose_landmarks:
                for landmark in landmark_points:
                    lm = results.pose_landmarks.landmark[landmark]
                    x, y = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # フレームをリストに追加
            extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 動画に書き込み
            out.write(frame)

    # リソースの解放
    cap.release()
    out.release()

    # === 切り取ったフレームを表示 ===
    st.subheader("📷 切り取ったフレーム")
    num_display = min(5, len(extracted_frames))  # 最大5枚表示
    for i in range(num_display):
        st.image(extracted_frames[i], caption=f"Frame {start_frame + i + 1}")

    # === 出力動画のダウンロード ===
    with open(output_video_path, "rb") as file:
        video_bytes = file.read()
    st.subheader("🎥 切り取った動画のダウンロード")
    st.download_button(label="📥 ダウンロード", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")
