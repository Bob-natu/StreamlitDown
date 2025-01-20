import streamlit as st
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

    # 動画の表示
    st.video(input_video_path)

    # 一時ディレクトリのクリーンアップ
    temp_dir.cleanup()
    st.info("一時ファイルをクリーンアップしました。")
