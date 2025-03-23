import streamlit as st
import numpy as np
import cv2

# Function to process image
def process_image(image, design, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Thresholding to detect the garment color
    mask_white = cv2.inRange(hsv, lower_color, upper_color)
    mask_black = cv2.bitwise_not(mask_white)

    # Convert to 3-channel masks
    mask_black_3CH = cv2.merge([mask_black, mask_black, mask_black])
    mask_white_3CH = cv2.merge([mask_white, mask_white, mask_white])

    # Apply masking
    dst3 = cv2.bitwise_and(mask_black_3CH, image)
    dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)

    # Resize design
    if design is not None:
        design = cv2.resize(design, (image.shape[1], image.shape[0]))
        design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)
        final_output = cv2.bitwise_and(design_mask_mixed, dst3_wh)
    else:
        final_output = dst3_wh

    return final_output

# Streamlit UI
st.title("2D Virtual Try-On")

uploaded_image = st.file_uploader("Upload a person image", type=["jpg", "png", "jpeg"])
uploaded_design = st.file_uploader("Upload a clothing image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    design = None
    if uploaded_design:
        design_bytes = np.asarray(bytearray(uploaded_design.read()), dtype=np.uint8)
        design = cv2.imdecode(design_bytes, cv2.IMREAD_COLOR)
        design = cv2.cvtColor(design, cv2.COLOR_BGR2RGB)

    # Define color thresholds for garment detection
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])

    # Process Image
    output_image = process_image(image, design, lower_green, upper_green)

    # Display Results
    st.image(image, caption="Original Image", use_container_width=True)
    st.image(output_image, caption="Virtual Try-On Output", use_container_width=True)
