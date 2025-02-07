import cv2
import mediapipe as mp
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

    # 出力動画の設定（保存用）
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 切り取ったフレームのリスト
    extracted_frames = []

    # 動画フレームを読み込み、処理する
    frame_count = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 文字描画用の設定
            frame_count += 1
            text = f"Frame: {frame_count}"

            # フレームに文字を中央揃えで描画
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2

            cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 画像を中央揃えにする
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 骨格描画（必要に応じて）
            if results.pose_landmarks:
                for landmark in mp_pose.POSE_CONNECTIONS:
                    x = int(results.pose_landmarks.landmark[landmark[0]].x * width)
                    y = int(results.pose_landmarks.landmark[landmark[0]].y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # フレームをリストに追加
            extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 動画に書き込み
            out.write(frame)

    # リソースの解放
    cap.release()
    out.release()

    # === 切り取ったフレームを表示 ===
    st.subheader("切り取ったフレーム")
    num_display = min(5, len(extracted_frames))  # 最大5枚表示
    for i in range(num_display):
        st.image(extracted_frames[i], caption=f"Frame {i + 1}")

    # === 出力動画のダウンロード ===
    with open(output_video_path, "rb") as file:
        video_bytes = file.read()
    st.subheader("切り取った動画のダウンロード")
    st.download_button(label="ダウンロード", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")
