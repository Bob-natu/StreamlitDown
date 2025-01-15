import streamlit as st
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

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
        output_graph_path = os.path.join(temp_dir.name, "output_graph.jpg")
        output_wrist_image_path = os.path.join(temp_dir.name, "highest_wrist.jpg")

        # MediaPipe Pose 初期化
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # データ保存用
        frame_numbers = []
        right_shoulder_y = []
        left_shoulder_y = []

        # 右手首と右肩の最高到達点データ
        highest_wrist_image = None
        min_right_wrist_y = float('inf')
        min_right_wrist_frame = -1
        min_right_shoulder_y = float('inf')
        highest_shoulder_frame = -1

        # 動画読み込み
        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # プロットのサイズを設定
        plot_width = 300
        plot_height = frame_height

        # 合成動画設定
        combined_width = frame_width + plot_width
        combined_height = max(frame_height, plot_height)
        fourcc = cv2.VideoWriter_fourcc(*'acv1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (combined_width, combined_height))

        # 解析中のメッセージを表示
        progress_bar = st.progress(0)
        st.info("動画を解析中です。しばらくお待ちください...")

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
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                    # データ保存
                    frame_numbers.append(frame_number)
                    right_shoulder_y.append(right_shoulder.y)
                    left_shoulder_y.append(left_shoulder.y)

                    # 右肩の最高到達点を記録
                    if right_shoulder.y < min_right_shoulder_y:
                        min_right_shoulder_y = right_shoulder.y
                        highest_shoulder_frame = frame_number

                    # 右手首の最高到達点を記録
                    if right_wrist.y < min_right_wrist_y:
                        min_right_wrist_y = right_wrist.y
                        min_right_wrist_frame = frame_number
                        highest_wrist_image = frame.copy()

                    # 骨格を描画
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # グラフ作成
                fig, ax = plt.subplots(figsize=(3, plot_height / 100), dpi=100)
                ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
                ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")
                ax.scatter([highest_shoulder_frame], [1 - min_right_shoulder_y], color='red', label="Highest Right Shoulder")
                ax.legend()
                ax.set_xlim(0, max(total_frames, frame_number + 10))
                ax.set_ylim(0, 1)
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Normalized Y Coordinate")
                plt.tight_layout()

                # グラフを画像として保存
                fig.canvas.draw()
                plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
                plt.close(fig)

                # グラフ画像とフレームを横に連結
                plot_image_resized = cv2.resize(plot_image, (plot_width, plot_height))
                combined_frame = np.hstack((frame, plot_image_resized))

                # 合成フレームを保存
                out.write(combined_frame)

                # 進捗バー更新
                progress = int((frame_number / total_frames) * 100)
                progress_bar.progress(progress)

            cap.release()
            out.release()

        # 解析完了メッセージ
        progress_bar.empty()
        st.success("解析が完了しました！")

        # 出力動画の表示
        st.video(output_video_path)

        # 最終的なグラフを作成して表示
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
        ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")
        ax.scatter([highest_shoulder_frame], [1 - min_right_shoulder_y], color='red', label="Highest Right Shoulder")
        ax.legend()
        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Normalized Y Coordinate")
        ax.set_title("Shoulder and Wrist Positions Over Time")
        plt.tight_layout()

        st.pyplot(fig)

        if highest_wrist_image is not None:
            # 右手首の最高到達点を中央に表示
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            st.image(highest_wrist_image, caption=f"右手首の最高到達点 (フレーム: {min_right_wrist_frame})", use_container_width=True, channels="BGR")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 右手首の最高到達点が見つからない場合
            st.markdown('<div class="centered"><p>右手首の最高到達点を特定できませんでした。</p></div>', unsafe_allow_html=True)

    finally:
        # クリーンアップを実行
        temp_dir.cleanup()
        st.info("一時ファイルをクリーンアップしました。")
