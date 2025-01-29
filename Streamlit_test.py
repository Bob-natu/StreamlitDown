import streamlit as st
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os

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
    output_graph_path = os.path.join(output_dir, "graph.jpg")
    output_shoulder_image_path = os.path.join(output_dir, "highest_wrist.jpg")

    # MediaPipe Pose 初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # データ保存用
    frame_numbers = []
    right_shoulder_y = []
    left_shoulder_y = []
    shoulder_angles = []

    # 右肩の最高到達点データ
    highest_shoulder_image = None
    highest_shoulder_landmarks = None
    min_right_shoulder_y = float('inf')  # 右肩の最小Y座標（最高到達点）
    min_right_shoulder_frame = -1  # 最高到達点のフレーム番号
    shoulder_angle_at_highest = None  # 最高到達点時の肩の角度

    # 動画読み込み
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けません: {input_video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        st.error(f"出力動画を作成できません: {output_video_path}")
        st.stop()

    # グラフの初期設定
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line_right, = ax.plot([], [], label="Right Shoulder Y", color="blue")
    line_left, = ax.plot([], [], label="Left Shoulder Y", color="green")
    highest_point, = ax.plot([], [], 'ro', label="Highest Right Shoulder")
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Y Coordinate (Flipped)")
    ax.set_title("Shoulder Coordinates Over Time")
    ax.legend()

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
                deel_angle = 180.0 - angle

                # 角度が 180° を超えた場合、負の値に変換
                if deel_angle > 180.0:
                    deel_angle -= 360.0

                shoulder_angles.append(deel_angle)

                # 右肩の最高到達点を記録
                if right_shoulder.y < min_right_shoulder_y:  # 最小Y座標が最高到達点
                    min_right_shoulder_y = right_shoulder.y
                    min_right_shoulder_frame = frame_number
                    highest_shoulder_image = frame.copy()
                    highest_shoulder_landmarks = results.pose_landmarks
                    shoulder_angle_at_highest = deel_angle

                # 骨格を描画
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 動画にフレーム番号と肩の角度をオーバーレイ表示
                cv2.putText(frame, f"Frame: {frame_number}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Shoulder Angle: {deel_angle:.2f} deg", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # グラフを更新
            line_right.set_data(frame_numbers, [1 - y for y in right_shoulder_y])  # 反転させることで頂点表示
            line_left.set_data(frame_numbers, [1 - y for y in left_shoulder_y])
            if min_right_shoulder_frame > 0:  # 右肩の最高到達点
                highest_point.set_data([min_right_shoulder_frame], [1 - min_right_shoulder_y])
            ax.set_xlim(0, max(10, frame_number + 10))
            plt.pause(0.001)

            # 動画保存
            out.write(frame)
            
        # リソース解放
        cap.release()
        out.release()

    # 動画ダウンロードボタン
    with open(output_video_path, "rb") as f:
        st.download_button("動画をダウンロード", f, file_name="processed_video.mp4", mime="video/mp4")

    # グラフ保存
    plt.ioff()
    if frame_numbers:
        plt.savefig(output_graph_path)
        st.pyplot(fig)  # グラフをStreamlitに表示
        plt.close()

    # 右肩の最高到達点画像保存（骨格描画付き）
    if highest_shoulder_image is not None and highest_shoulder_landmarks is not None:
        try:
            # 骨格を画像に描画
            mp_drawing.draw_landmarks(highest_shoulder_image, highest_shoulder_landmarks, mp_pose.POSE_CONNECTIONS)

            # 画像にフレーム番号と肩の角度をオーバーレイ表示
            cv2.putText(highest_shoulder_image, f"Highest Frame: {min_right_shoulder_frame}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            if shoulder_angle_at_highest is not None:
                cv2.putText(highest_shoulder_image, f"Shoulder Angle: {shoulder_angle_at_highest:.2f} deg", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 画像保存
            success = cv2.imwrite(output_shoulder_image_path, highest_shoulder_image)
            if success:
                st.image(highest_shoulder_image, channels="BGR", use_container_width=True)  # 画像を表示
            else:
                st.error(f"画像保存に失敗しました: {output_shoulder_image_path}")
        except Exception as e:
            st.error(f"右肩画像保存中にエラーが発生しました: {e}")
    else:
        st.warning("最高到達点の画像が存在しないか、骨格データがありません。")
