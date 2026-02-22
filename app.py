import webbrowser
import sys
import time

if getattr(sys, 'frozen', False):
    time.sleep(3)           # wait for server to start
    webbrowser.open_new("http://localhost:8501")
import streamlit as st
import cv2
import numpy as np
import time

# ================================================
# FIXED: Define display sizes RIGHT HERE at the top
# ================================================
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

# ================================================
# App Title & Intro
# ================================================
st.title("Underwater Video Enhancer")
st.markdown("""
Upload a video or use your live camera to see real-time enhancement using color correction and CLAHE.
Adjust sliders to tune contrast and color boost.
""")

# ================================================
# Sidebar Controls
# ================================================
st.sidebar.header("Enhancement Settings")

clip_limit = st.sidebar.slider(
    "CLAHE Clip Limit (contrast strength)",
    min_value=0.5,
    max_value=6.0,
    value=2.5,
    step=0.1,
    help="Higher values = stronger contrast enhancement"
)

color_boost = st.sidebar.slider(
    "Color Boost Factor",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Higher values = stronger correction of blue-green tint"
)

display_mode = st.sidebar.radio(
    "Display Mode",
    ["Side-by-Side", "Enhanced Only"]
)

# Input selection
input_option = st.radio(
    "Input Source",
    ["Upload Video", "Live Camera"]
)

# Placeholders for display & messages
frame_placeholder = st.empty()
status_text = st.empty()

# ================================================
# Core Enhancement Function
# ================================================
def enhance_frame(frame, clip_limit, color_boost):
    # Color correction
    frame_float = frame.astype(np.float32) / 255.0
    b, g, r = cv2.split(frame_float)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    max_mean = max(mean_b, mean_g, mean_r)

    r_corrected = np.clip(r * (color_boost * max_mean / (mean_r + 1e-6)), 0, 1)
    g_corrected = np.clip(g * (color_boost * max_mean / (mean_g + 1e-6)), 0, 1)
    b_corrected = b

    corrected = cv2.merge([b_corrected, g_corrected, r_corrected])

    # CLAHE contrast
    corrected_uint8 = (corrected * 255).astype(np.uint8)
    lab = cv2.cvtColor(corrected_uint8, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b_lab])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

# ================================================
# Upload Video Processing
# ================================================
if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = "temp_upload.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            st.error("Could not open uploaded video.")
        else:
            status_text.success("Video loaded! Processing...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                enhanced = enhance_frame(frame, clip_limit, color_boost)

                original_small = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                enhanced_small = cv2.resize(enhanced, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                if display_mode == "Side-by-Side":
                    display = np.hstack((original_small, enhanced_small))
                else:
                    display = enhanced_small

                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display_rgb, channels="RGB", use_column_width=True)

                time.sleep(0.03)  # Simulate real-time

            cap.release()
            status_text.success("Processing complete!")

# ================================================
# Live Camera Processing
# ================================================
else:
    st.info("Starting live camera... Allow camera access in browser")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not detected. Try index 1 or 2, or check permissions.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            enhanced = enhance_frame(frame, clip_limit, color_boost)

            original_small = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            enhanced_small = cv2.resize(enhanced, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            if display_mode == "Side-by-Side":
                display = np.hstack((original_small, enhanced_small))
            else:
                display = enhanced_small

            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_rgb, channels="RGB", use_column_width=True)

            if st.button("Stop Live"):
                break

            time.sleep(0.03)

        cap.release()