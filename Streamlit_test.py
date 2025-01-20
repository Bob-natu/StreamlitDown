import streamlit as st
import cv2
import mediapipe as mp
import os
import tempfile

# 動画ファイルをアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 一時ディレクトリ作成
    temp_dir = tempfile.TemporaryDirectory()
    input_video_path = os.path.join(temp_dir.name, uploaded_file.name)

    # アップロードされた動画を保存
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("動画がアップロードされました。解析を開始します。")

    # MediaPipe Pose 初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 動画読み込み
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力動画の設定
    output_video_path = os.path.join(temp_dir.name, "output_video_with_pose.mp4")  # 一時ディレクトリに保存
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4形式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Pose インスタンス作成
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # フレームをRGBに変換して処理
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # 骨格描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 骨格を描画したフレームを出力動画に書き込み
            out.write(frame)

        cap.release()
        out.release()

    if os.path.exists(output_video_path):
        st.success("動画が正しく保存されました！")
    else:
        st.error("動画が保存されていません。")
    
    st.success("骨格抽出が完了しました！")
    st.write("動画の保存パス:", output_video_path)

    # 動画再生
    st.video("output_video_with_pose.mp4")

    # 一時ディレクトリのクリーンアップ
    temp_dir.cleanup()
    st.info("一時ファイルをクリーンアップしました。")
