import streamlit as st
import numpy as np
import cv2

# Streamlit app configuration
st.set_page_config(page_title="2D Virtual Try-On App", layout="wide")
st.title("👕 2D Virtual Clothing Try-On")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    person_img = st.file_uploader("Upload person image", type=["png", "jpg", "jpeg"])
with col2:
    design_img = st.file_uploader("Upload clothing design", type=["png", "jpg", "jpeg"])

if person_img is not None and design_img is not None:
    # Read images
    file_bytes = np.asarray(bytearray(person_img.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    design_bytes = np.asarray(bytearray(design_img.read()), dtype=np.uint8)
    design = cv2.imdecode(design_bytes, cv2.IMREAD_COLOR)

    # Original image processing code (unchanged)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges (excluding skin tones)
    color_ranges = {
        "green": ([25, 52, 72], [102, 255, 255]),
        "blue": ([90, 50, 50], [130, 255, 255]),
        "red1": ([0, 150, 100], [10, 255, 255]), 
        "red2": ([170, 150, 100], [180, 255, 255]),
        "yellow": ([20, 150, 150], [30, 255, 255]),
        "orange": ([10, 180, 100], [25, 255, 255]),
        "purple": ([130, 50, 50], [160, 255, 255]),
        "pink": ([160, 100, 100], [180, 255, 255]),
    }

    # Create an empty mask
    mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Apply masks for all defined colors
    for lower, upper in color_ranges.values():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_total = cv2.bitwise_or(mask_total, mask)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)

    # Ignore small areas
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            cv2.drawContours(mask_cleaned, [cnt], -1, 0, -1)

    # Prepare design
    design = cv2.resize(design, (frame.shape[1], frame.shape[0]))

    # Ensure the design only appears where the mask is
    design_masked = cv2.bitwise_and(design, design, mask=mask_cleaned)

    # Invert mask for blending
    mask_black = cv2.bitwise_not(mask_cleaned)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask_black)

    # Combine design and original frame
    final_output = cv2.addWeighted(frame_masked, 1, design_masked, 1, 0)

    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

    # Display results in two columns
    st.subheader("Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(frame_rgb, caption="Original Image", use_container_width=True)
    
    with col2:
        st.image(output_rgb, caption="Virtual Try-On Result", use_container_width=True)

elif person_img is not None or design_img is not None:
    st.warning("Please upload both images to see the virtual try-on result")
else:
    st.info("Upload a person image and a clothing design to get started")

# Add some instructions
st.markdown("---")
st.subheader("Instructions")
st.markdown("""
1. Upload a photo of a person wearing solid-colored clothing
2. Upload the clothing design you want to try on
3. See your virtual try-on result!
""")