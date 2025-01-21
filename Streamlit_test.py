import streamlit as st
import cv2
import os
import tempfile
import mediapipe as mp
import shutil

# 動画アップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 一時ディレクトリ作成
    temp_dir = tempfile.TemporaryDirectory()

    # 一時ファイルに保存
    input_video_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 出力ファイルのパス
    output_video_path = os.path.join(temp_dir.name, 'output_video_with_pose.mp4')  # 修正箇所
    # MediaPipe Pose 初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 動画読み込み
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # mp4vを使用
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Pose インスタンス作成
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 骨格抽出処理
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # 骨格を描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 処理したフレームを書き込む
            out.write(frame)

        cap.release()
        out.release()

    # 処理が完了した動画を表示
    st.success("動画の処理が完了しました！")

    # Streamlitでメモリ上のファイルを表示するため、ファイルを読み込む
    with open(output_video_path, "rb") as video_file:
        video_bytes = video_file.read()

    # メモリから動画を読み込み、Streamlitで表示
    st.video(video_bytes)

    # 一時ディレクトリのクリーンアップ
    temp_dir.cleanup()
