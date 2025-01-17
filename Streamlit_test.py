import streamlit as st
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
import os
import tempfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # 修正

# CSS を使用して画像を中央ぞろえにする
st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Matplotlibのバックエンドを明示的に設定
import matplotlib
matplotlib.use("Agg")  # 非インタラクティブモードを指定

# Streamlit UI設定
st.write("## 動画解析: 手首と肩の位置プロット")
st.sidebar.header("設定")

# ファイルアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 一時ディレクトリ作成
    temp_dir = tempfile.TemporaryDirectory()
    try:
        input_video_path = os.path.join(temp_dir.name, uploaded_file.name)

        # アップロードされた動画を保存
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success("動画がアップロードされました。解析を開始します。")

        # 出力ファイルパス設定
        output_video_path = os.path.join(temp_dir.name, "output_video_with_plot.mp4")

        # MediaPipe Pose 初期化
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # データ保存用
        frame_numbers = []
        right_shoulder_y = []
        left_shoulder_y = []

        # 動画読み込み
        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 合成動画設定
        combined_width = frame_width + 300
        combined_height = frame_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (combined_width, combined_height))

        # Pose インスタンス作成
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # フレームをRGBに変換して処理
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

                    # データ保存
                    frame_numbers.append(frame_number)
                    right_shoulder_y.append(right_shoulder.y)
                    left_shoulder_y.append(left_shoulder.y)

                    # 骨格を描画
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # グラフ作成
                fig, ax = plt.subplots(figsize=(3, frame_height / 100), dpi=100)
                ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
                ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")
                ax.legend()
                ax.set_xlim(0, total_frames)
                ax.set_ylim(0, 1)
                plt.tight_layout()

                # グラフを画像として保存
                canvas = FigureCanvas(fig)
                canvas.draw()
                plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)  # 修正
                plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                # グラフ画像とフレームを横に連結
                plot_image_resized = cv2.resize(plot_image, (300, frame_height))
                combined_frame = np.hstack((frame, plot_image_resized))

                # 合成フレームを保存
                out.write(combined_frame)

            cap.release()
            out.release()

        st.success("解析が完了しました！")
        st.video(output_video_path)

    finally:
        temp_dir.cleanup()
        st.info("一時ファイルをクリーンアップしました。")
