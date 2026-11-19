"""Privacy-safe Streamlit interface for image detection with a local YOLO weight."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource(show_spinner=False)
def load_detector(weight_path: str):
    return YOLO(weight_path)


def main():
    st.set_page_config(page_title="Road-scene detector", layout="wide")
    st.title("公交道路场景识别演示")
    st.caption("图片仅在当前运行环境中处理；公开仓库不提供私有数据或模型权重。")

    weight_path = st.sidebar.text_input("YOLO 权重路径", "weights/best.pt")
    confidence = st.sidebar.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
    upload = st.file_uploader("上传一张测试图片", type=["jpg", "jpeg", "png"])
    if upload is None:
        return
    image = Image.open(upload).convert("RGB")
    left, right = st.columns(2)
    left.image(image, caption="输入图像", use_container_width=True)
    if not Path(weight_path).is_file():
        right.warning("请在侧栏填写本机已有的权重路径。")
        return
    with st.spinner("正在推理…"):
        result = load_detector(weight_path).predict(
            source=np.asarray(image), conf=confidence, verbose=False
        )[0]
    right.image(result.plot()[:, :, ::-1], caption="检测结果", use_container_width=True)
    st.write(f"检测到 {len(result.boxes)} 个目标")


if __name__ == "__main__":
    main()

