import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import base64

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)
    if not cap.isOpened():
        st.error("動画の読み込みに失敗しました。")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out_file:
        output_processed_video_path = temp_out_file.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_processed_video_path, fourcc, 30.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()

    with open(output_processed_video_path, "rb") as video_file:
        video_bytes = video_file.read()

    st.write(f"Base64エンコードされた動画データの長さ: {len(video_bytes)} バイト")

    # HTML埋め込みの修正版
    video_data_url = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
    st.markdown(
        f'<video width="600" controls><source src="{video_data_url}" type="video/mp4"></video>',
        unsafe_allow_html=True
    )

else:
    st.info("動画ファイルをアップロードしてください。")
