import streamlit as st
import cv2
import mediapipe as mp
import math
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
    highest_shoulder_y = float('inf')  # 右肩の最高点
    highest_frame_number = -1  # 最高点に達したフレーム番号

    # 動画読み込み
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けません: {input_video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # グラフの初期設定
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#f0f0f0')  # グレー背景
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Y Coordinate (Flipped)", fontsize=12)
    ax.set_title("Shoulder Coordinates Over Time", fontsize=14)
    ax.legend(["Right Shoulder Y", "Left Shoulder Y", "Highest Right Shoulder"])
    
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

                # 両肩のY座標をリストに保存
                frame_numbers.append(frame_number)
                right_shoulder_y.append(right_shoulder.y)
                left_shoulder_y.append(left_shoulder.y)

                # 右肩の最高到達点を記録
                if right_shoulder.y < highest_shoulder_y:  # Y座標が最小（最高到達点）
                    highest_shoulder_y = right_shoulder.y
                    highest_frame_number = frame_number

                # 骨格を描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # グラフを更新
            ax.clear()
            ax.set_facecolor('#f0f0f0')
            ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue", linewidth=2)
            ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green", linewidth=2)

            if highest_frame_number != -1:
                ax.plot([highest_frame_number], [1 - highest_shoulder_y], 'ro', label="Highest Right Shoulder", markersize=8)

            ax.set_xlim(0, max(10, frame_number + 10))
            ax.set_ylim(0, 1)
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_ylabel("Y Coordinate (Flipped)", fontsize=12)
            ax.set_title("Shoulder Coordinates Over Time", fontsize=14)
            ax.legend(fontsize=10)

            # グラフを更新
            plt.pause(0.001)

            # グラフを画像に変換
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # バッファをNumPy配列として読み込む
            graph_image = np.array(bytearray(buf.read()), dtype=np.uint8)
            graph_image = cv2.imdecode(graph_image, cv2.IMREAD_COLOR)

            # グラフを動画フレームの下部に結合
            graph_resized = cv2.resize(graph_image, (frame_width, 150))  # グラフを150ピクセルに調整
            frame_with_graph = np.vstack([frame, graph_resized])  # 縦に結合

            # 動画保存
            out.write(frame_with_graph)
        
        # リソース解放
        cap.release()
        out.release()

    # 処理した動画をアプリ内で表示
    st.video(output_video_path)

    # 動画ダウンロードボタン
    with open(output_video_path, "rb") as f:
        st.download_button("動画をダウンロード", f, file_name="processed_video.mp4", mime="video/mp4")
