from pathlib import Path
import sys, tempfile, os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import streamlit as st
import pandas as pd
import plotly.express as px
from src.detector import TrafficDetector
from src.traffic_analysis import build_record, summarize_results
from src.video_utils import open_video, get_video_metadata

st.set_page_config(page_title="Intelligent Traffic Monitoring", page_icon="🚦", layout="wide")
st.title("🚦 Intelligent Traffic Monitoring & Congestion Analysis")
st.caption("Computer vision prototype • Vehicle detection and sampled-frame congestion indicators")

with st.sidebar:
    st.header("Analysis settings")
    model_name = st.text_input("YOLO model weights", "yolov8n.pt")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    frame_interval = st.slider("Analyze every Nth frame", 1, 60, 15)
    low_threshold = st.number_input("Low congestion upper threshold (vehicles/frame)", 0, 100, 5)
    high_threshold = st.number_input("High congestion threshold (vehicles/frame)", 1, 200, 12)
    st.warning("Use only traffic footage you are authorized to process. Avoid uploading identifiable or sensitive footage.")

st.info("Congestion is estimated from vehicle counts in sampled frames. Calibrate thresholds for the camera view; this is not a validated traffic-flow forecast.")

uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    if high_threshold <= low_threshold:
        st.error("High threshold must be greater than low threshold.")
        st.stop()

    st.video(uploaded)
    if st.button("Analyze traffic video", type="primary"):
        temp_path = None
        try:
            suffix = "." + uploaded.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = tmp.name

            detector = TrafficDetector(model_name=model_name, confidence=confidence)
            cap = open_video(temp_path)
            meta = get_video_metadata(cap)
            st.write(f"Video: {meta['width']} × {meta['height']} | {meta['fps']:.2f} FPS | {meta['frame_count']} frames")

            records = []
            preview = None
            progress = st.progress(0)
            status = st.empty()
            frame_idx = 0
            total_frames = max(meta["frame_count"], 1)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_interval == 0:
                    annotated, counts, detections = detector.detect(frame)
                    timestamp = frame_idx / meta["fps"] if meta["fps"] > 0 else 0
                    records.append(build_record(frame_idx, timestamp, counts, low_threshold, high_threshold))
                    if preview is None:
                        preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))
                    status.write(f"Processed {frame_idx:,} / {total_frames:,} frames")

            cap.release()
            progress.progress(1.0)
            status.success("Video analysis completed.")
            result_df = summarize_results(records)

            if result_df.empty:
                st.warning("No frames were analyzed.")
            else:
                st.session_state["traffic_results"] = result_df
                st.session_state["traffic_preview"] = preview
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if "traffic_results" in st.session_state:
    df = st.session_state["traffic_results"]
    st.subheader("Analysis results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames analyzed", f"{len(df):,}")
    c2.metric("Average vehicles / sampled frame", f"{df['total_vehicles'].mean():.1f}")
    most_common = df["congestion_level"].mode().iloc[0]
    c3.metric("Most frequent congestion label", most_common)

    if st.session_state.get("traffic_preview") is not None:
        st.image(st.session_state["traffic_preview"], caption="Annotated sample frame", use_container_width=True)

    st.plotly_chart(px.line(df, x="timestamp_seconds", y="total_vehicles",
                            title="Detected vehicles over sampled video time",
                            labels={"timestamp_seconds": "Time (seconds)", "total_vehicles": "Vehicles detected"}),
                    use_container_width=True)
    level_counts = df["congestion_level"].value_counts().rename_axis("Congestion").reset_index(name="Samples")
    st.plotly_chart(px.bar(level_counts, x="Congestion", y="Samples", title="Congestion label distribution"),
                    use_container_width=True)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download traffic analysis CSV", df.to_csv(index=False),
                       file_name="traffic_analysis.csv", mime="text/csv")
else:
    st.markdown("### What this prototype provides")
    st.markdown("- Vehicle detection for cars, motorcycles, buses and trucks\n- Sampled-frame vehicle counts\n- Configurable low/moderate/high congestion labels\n- Time-series visualization and CSV export")
