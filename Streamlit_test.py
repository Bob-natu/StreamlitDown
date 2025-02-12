import streamlit as st
import cv2
import mediapipe as mp
import os
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# Streamlit UI設定
st.title("肩の位置追跡とグラフ作成")

# 動画アップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # 動画の読み込み
    input_video_path = os.path.join("/tmp", uploaded_file.name)
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 出力設定
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "output.mp4")

    # MediaPipe Pose 初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # データ保存用
    frame_numbers = []
    right_shoulder_y = []
    left_shoulder_y = []
    right_wrist_y = []
    
    highest_shoulder_y = float('inf')  # 右肩の最高点
    highest_frame_number = -1  # 最高点に達したフレーム番号
    highest_wrist_y = float('inf')  # 右手首の最高点（最小Y座標）
    highest_wrist_frame = None  # 最高点フレーム

    # 動画読み込み
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けません: {input_video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 コーデックに変更
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Pose インスタンス作成
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.success("動画処理が完了しました。")
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # フレームをRGBに変換して処理
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # 両肩のY座標をリストに保存
                frame_numbers.append(frame_number)
                right_shoulder_y.append(right_shoulder.y)
                left_shoulder_y.append(left_shoulder.y)
                right_wrist_y.append(right_wrist.y)

                # 右肩の最高到達点を記録
                if right_shoulder.y < highest_shoulder_y:
                    highest_shoulder_y = right_shoulder.y
                    highest_frame_number = frame_number
                
                # 右手首の最高点を記録
                if right_wrist.y < highest_wrist_y:
                    highest_wrist_y = right_wrist.y
                    highest_wrist_frame = frame.copy()

                # 骨格を描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # フレームを書き込む
            out.write(frame)
    
    # リソース解放
    cap.release()
    out.release()

    # グラフを作成しアプリ内に表示
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#f0f0f0')
    ax.plot(frame_numbers, right_shoulder_y, label="Right Shoulder Y", color="blue")
    ax.plot(frame_numbers, left_shoulder_y, label="Left Shoulder Y", color="green")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Shoulder Position Over Time")
    ax.legend()
    st.pyplot(fig)
    
    # 右手首の最高点のフレームを表示
    if highest_wrist_frame is not None:
        highest_wrist_frame_rgb = cv2.cvtColor(highest_wrist_frame, cv2.COLOR_BGR2RGB)
        st.image(highest_wrist_frame_rgb, caption="右手首が最も高いフレーム", use_container_width=True)

    # 動画ダウンロードボタン
    with open(output_video_path, "rb") as f:
        st.download_button("動画をダウンロード", f, file_name="processed_video.mp4", mime="video/mp4")
