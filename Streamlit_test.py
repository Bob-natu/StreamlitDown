import streamlit as st
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import io

# Streamlit アプリの設定
st.title("動画解析: 手首と肩の位置プロット")
st.sidebar.header("設定")

# ファイルアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 一時入力ファイルの作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        temp_input_file.flush()

    st.sidebar.success("動画がアップロードされました。解析を開始します。")

    # MediaPipe Poseの設定
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # データ保存用
    frame_numbers = []
    right_shoulder_y = []
    left_shoulder_y = []
    min_right_wrist_y = float('inf')
    highest_wrist_image = None

    # 動画の読み込み
    cap = cv2.VideoCapture(temp_input_file.name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # デバッグ用メッセージ
    st.write(f"動画の解像度: {frame_width}x{frame_height}, フレーム数: {total_frames}, FPS: {fps}")

    # 一時出力ファイルの作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
        output_path = temp_output_file.name

    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264が使用できない場合の代替コーデック
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        st.error("VideoWriterの初期化に失敗しました。コーデックまたは出力パスを確認してください。")
    else:
        # 進捗バー
        progress_bar = st.progress(0)

        # Poseインスタンスの作成
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # フレームをRGBに変換してMediaPipeで処理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                # フレーム番号の取得
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                    # データの記録
                    frame_numbers.append(frame_number)
                    right_shoulder_y.append(right_shoulder.y)
                    left_shoulder_y.append(left_shoulder.y)

                    # 手首の最高点の記録
                    if right_wrist.y < min_right_wrist_y:
                        min_right_wrist_y = right_wrist.y
                        highest_wrist_image = frame.copy()

                    # 骨格の描画
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # フレームの保存
                out.write(frame)

                # 進捗バーの更新
                progress = int((frame_number / total_frames) * 100)
                progress_bar.progress(min(progress, 100))

            # リソース解放
            cap.release()
            out.release()

            # 動画解析完了
            st.success("解析が完了しました！")
            progress_bar.empty()

        # 保存された動画をメモリに読み込む
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()

        # 出力動画の表示
        st.video(io.BytesIO(video_bytes))

        # 肩と手首の位置データのグラフ化
        if frame_numbers and right_shoulder_y and left_shoulder_y:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
            ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Normalized Y Coordinate")
            ax.set_title("Shoulder and Wrist Positions Over Time")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("グラフデータが記録されていません。動画を確認してください。")

        # 右手首の最高到達点の画像表示
        if highest_wrist_image is not None:
            st.image(
                highest_wrist_image,
                caption="右手首の最高到達点",
                use_container_width=True,
                channels="BGR"
            )
        else:
            st.warning("右手首の最高到達点が見つかりませんでした。")
