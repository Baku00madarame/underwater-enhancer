import streamlit as st
import cv2
import numpy as np
import time

# Display sizes for resized preview
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

# Title
st.title("Underwater Video Enhancer")
st.markdown("""
Upload a video or use live camera to see real-time enhancement with color correction and CLAHE.
Adjust sliders to tune results.
""")

# Sidebar controls
st.sidebar.header("Enhancement Settings")

clip_limit = st.sidebar.slider(
    "CLAHE Clip Limit (contrast strength)",
    0.5, 6.0, 2.5, 0.1,
    help="Higher values increase contrast but may add noise"
)

color_boost = st.sidebar.slider(
    "Color Boost Factor",
    0.5, 3.0, 1.0, 0.1,
    help="Higher values strengthen red/green correction"
)

display_mode = st.sidebar.radio(
    "Display Mode",
    ["Side-by-Side", "Enhanced Only"]
)

input_option = st.radio(
    "Input Source",
    ["Upload Video", "Live Camera"]
)

# Placeholders
frame_placeholder = st.empty()
status_text = st.empty()
progress_bar = st.progress(0)

# Enhancement function
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

# Upload Video – Fixed version with progress bar & skipped frames
if input_option == "Upload Video":
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        help="MP4 recommended. Keep file <50MB for fast processing"
    )

    if uploaded_file is not None:
        temp_path = "temp_upload.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Could not open video.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            status_text.success(f"Video loaded! {total_frames} frames total.")

            progress_bar.progress(0)
            current_frame_placeholder = st.empty()

            frame_index = 0
            processed_frames = 0
            skip_frames = 5  # Process every 5th frame to avoid flicker & timeout

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % skip_frames == 0:
                    enhanced = enhance_frame(frame, clip_limit, color_boost)

                    original_small = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    enhanced_small = cv2.resize(enhanced, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                    if display_mode == "Side-by-Side":
                        display = np.hstack((original_small, enhanced_small))
                    else:
                        display = enhanced_small

                    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    current_frame_placeholder.image(display_rgb, channels="RGB", use_column_width=True)

                    processed_frames += 1

                # Update progress bar
                progress = (frame_index + 1) / total_frames
                progress_bar.progress(progress)

                frame_index += 1

                if frame_index > 1200:  # Safety limit (~40s at 30fps)
                    status_text.warning("Long video – preview stopped after 1200 frames.")
                    break

            cap.release()
            status_text.success(f"Preview complete! Processed {processed_frames} frames (every {skip_frames}th).")

# Live Camera
else:
    st.info("Starting live camera... Allow browser camera access.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not detected. Try changing to index 1 or check permissions.")
    else:
        status_text.success("Live camera connected! Real-time enhancement active.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame.")
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

            if st.button("Stop Live Feed"):
                status_text.info("Live feed stopped.")
                break

            time.sleep(0.03)

        cap.release()
