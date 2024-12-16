import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image

# Define the path to your folder where the models are stored
MODEL_DIR = r"C:\Users\Lenovo\Downloads\Image-Inpainting\colorization_models"
PROTOTXT = os.path.join(MODEL_DIR, 'colorization_deploy_v2.prototxt')
POINTS = os.path.join(MODEL_DIR, 'pts_in_hull.npy')
MODEL = os.path.join(MODEL_DIR, 'colorization_release_v2.caffemodel')

# Load the colorization model
def load_colorization_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

# Function to colorize the image
def colorize_image(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

    net = load_colorization_model()
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")


# Function to reduce noise in an image
def reduce_noise(image, method="gaussian"):
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(image, 5)
    else:
        raise ValueError("Method not recognized. Choose 'gaussian' or 'median'.")

# Function to enhance sharpness
def enhance_sharpness(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Function to remove scratches automatically
def remove_scratches(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    result = cv2.inpaint(image, dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return result

# Function to remove haze from an image
def dehaze_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge((l, a, b))
    dehazed_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return dehazed_image

# Function to enhance contrast
def enhance_contrast(image):
    alpha = 1.5  # Contrast control
    beta = 20    # Brightness control
    contrast_enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_enhanced

# Streamlit UI
st.title("Historical Image Restoration Tool")
st.write("Upload an image and apply different transformations.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL and convert to numpy array
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Select operation
    operation = st.selectbox(
        "Choose an operation",
        ("Colorize", "Reduce Noise", "Enhance Sharpness", "Remove Scratches", "Dehaze", "Enhance Contrast")
    )

    if operation == "Colorize":
        colorized_image = colorize_image(image)
        st.image(colorized_image, caption="Colorized Image")

    elif operation == "Reduce Noise":
        denoised_image = reduce_noise(image, method="median")
        st.image(denoised_image, caption="Denoised Image")

    elif operation == "Enhance Sharpness":
        sharpened_image = enhance_sharpness(image)
        st.image(sharpened_image, caption="Sharpened Image")

    elif operation == "Remove Scratches":
        scratched_removed_image = remove_scratches(image)
        st.image(scratched_removed_image, caption="Scratches Removed Image")

    elif operation == "Dehaze":
        dehazed_image = dehaze_image(image)
        st.image(dehazed_image, caption="Dehazed Image")

    elif operation == "Enhance Contrast":
        contrast_enhanced_image = enhance_contrast(image)
        st.image(contrast_enhanced_image, caption="Contrast Enhanced Image")
