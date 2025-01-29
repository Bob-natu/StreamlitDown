import streamlit as st
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt

# MediaPipe のセットアップ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 動画ファイルのアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 動画を読み込む
    cap = cv2.VideoCapture(uploaded_file.name)

    # グラフデータ用リスト
    frame_numbers = []
    right_shoulder_y = []
    left_shoulder_y = []
    shoulder_angles = []

    # 右肩の最高到達点
    min_right_shoulder_y = float('inf')
    min_right_shoulder_frame = -1

    # Streamlit でリアルタイム更新のための準備
    graph_placeholder = st.empty()  # グラフを更新するためのプレースホルダー
    video_placeholder = st.empty()  # 動画のプレビュー用プレースホルダー

    # 動画処理ループ
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

            # 両肩のY座標保存
            frame_numbers.append(frame_number)
            right_shoulder_y.append(right_shoulder.y)
            left_shoulder_y.append(left_shoulder.y)

            # 肩の角度を計算
            left_x, left_y = left_shoulder.x, left_shoulder.y
            right_x, right_y = right_shoulder.x, right_shoulder.y
            delta_x = left_x - right_x
            delta_y = left_y - right_y
            angle = math.degrees(math.atan2(delta_y, delta_x))
            shoulder_angles.append(angle)

            # 右肩の最高到達点を記録
            if right_shoulder.y < min_right_shoulder_y:
                min_right_shoulder_y = right_shoulder.y
                min_right_shoulder_frame = frame_number

            # 骨格を描画
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 動画にフレーム番号をオーバーレイ表示
            cv2.putText(frame, f"Frame: {frame_number}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {angle:.2f} deg", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 動画のプレビューをリアルタイム表示
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # グラフをリアルタイムで描画・更新
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
        ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")

        # 最高点に赤いプロットを追加
        if min_right_shoulder_frame > 0:
            ax.plot(
                min_right_shoulder_frame, 
                1 - min_right_shoulder_y, 
                'ro',  # 赤い丸印
                label="Highest Right Shoulder"
            )
            ax.annotate(
                f"Highest Point\nFrame: {min_right_shoulder_frame}",
                xy=(min_right_shoulder_frame, 1 - min_right_shoulder_y),
                xytext=(min_right_shoulder_frame + 5, 1 - min_right_shoulder_y + 0.05),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10,
                color="red"
            )

        ax.set_xlim(0, max(10, frame_number + 10))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Y Coordinate (Flipped)")
        ax.set_title("Shoulder Coordinates Over Time")
        ax.legend()

        # グラフを Streamlit で表示
        graph_placeholder.pyplot(fig)
        plt.close(fig)

    # 処理が完了したらリソースを解放
    cap.release()
