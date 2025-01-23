import cv2
import mediapipe as mp
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
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 一時的に保存する処理後の動画ファイルの準備
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out_file:
        output_processed_video_path = temp_out_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_processed_video_path, fourcc, 30.0, (frame_width, frame_height))

    # 動画処理と骨格抽出
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipeで骨格を処理
        results = pose.process(image_rgb)

        # 骨格が検出された場合、ランドマークを描画
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 処理後のフレームを動画ファイルに書き込み
        out.write(frame)

    cap.release()
    out.release()

    # 処理後の動画をStreamlitで表示
    st.write(f"処理後の動画を表示します: {output_processed_video_path}")
    
    # 処理後の動画をバイナリモードで読み込む
    with open(output_processed_video_path, 'rb') as f:
        video_bytes = f.read()
    
    # バイナリデータで表示
    st.video(video_bytes)

else:
    st.info("動画ファイルをアップロードしてください。")
