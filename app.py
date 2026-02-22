import streamlit as st
import cv2
import numpy as np
import time

# Display sizes
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

st.title("Underwater Video Enhancer")
st.markdown("Upload a video or use live camera. Adjust sliders for real-time tuning.")

# Sidebar
st.sidebar.header("Settings")
clip_limit = st.sidebar.slider("CLAHE Clip Limit", 0.5, 6.0, 2.5, 0.1)
color_boost = st.sidebar.slider("Color Boost Factor", 0.5, 3.0, 1.0, 0.1)
display_mode = st.sidebar.radio("Display Mode", ["Side-by-Side", "Enhanced Only"])

input_option = st.radio("Input", ["Upload Video", "Live Camera"])

frame_placeholder = st.empty()
status_text = st.empty()
progress_bar = st.progress(0)

def enhance_frame(frame, clip_limit, color_boost):
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
    corrected_uint8 = (corrected * 255).astype(np.uint8)
    lab = cv2.cvtColor(corrected_uint8, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b_lab])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

# Upload Video – with Pause/Resume
if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Cannot open video")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            status_text.success(f"Video loaded – {total_frames} frames")

            # Controls
            col1, col2 = st.columns(2)
            pause_btn = col1.button("Pause" if 'playing' not in st.session_state else "Resume")
            stop_btn = col2.button("Stop")

            if 'playing' not in st.session_state:
                st.session_state.playing = True
                st.session_state.frame_index = 0

            if pause_btn:
                st.session_state.playing = not st.session_state.playing

            if stop_btn:
                st.session_state.playing = False
                st.session_state.frame_index = 0
                cap.release()
                st.experimental_rerun()

            if st.session_state.playing:
                progress_bar.progress(st.session_state.frame_index / total_frames)

                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_index)
                ret, frame = cap.read()

                if ret:
                    enhanced = enhance_frame(frame, clip_limit, color_boost)

                    original_small = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    enhanced_small = cv2.resize(enhanced, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                    if display_mode == "Side-by-Side":
                        display = np.hstack((original_small, enhanced_small))
                    else:
                        display = enhanced_small

                    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_rgb, channels="RGB", use_column_width=True)

                    st.session_state.frame_index += 1

                    # Auto pause at end
                    if st.session_state.frame_index >= total_frames:
                        st.session_state.playing = False
                        status_text.success("Video finished!")
                else:
                    st.session_state.playing = False
                    status_text.success("End of video.")

            cap.release()

# Live Camera (unchanged)
else:
    st.info("Live camera starting... Allow access.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("No camera detected.")
    else:
        status_text.success("Live feed active.")

        while True:
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

            if st.button("Stop Live"):
                break

            time.sleep(0.03)

        cap.release()
