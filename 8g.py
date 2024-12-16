import numpy as np
import cv2
import os

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

# Sketcher class for interactive mask creation
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])
        cv2.imshow(self.windowname + ": mask", self.dests[1])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()

# Function to perform inpainting interactively
def inpaint_image_interactive(image):
    img_mask = image.copy()
    inpaintMask = np.zeros(image.shape[:2], np.uint8)
    sketch = Sketcher("Interactive Inpainting", [img_mask, inpaintMask], lambda: ((255, 255, 255), 255))

    while True:
        ch = cv2.waitKey()
        if ch == 27:  # ESC key
            break
        if ch == ord('t'):
            res = cv2.inpaint(image, inpaintMask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            cv2.imshow('Inpaint Output using FMM', res)
            return res
        if ch == ord('r'):
            img_mask[:] = image
            inpaintMask[:] = 0
            sketch.show()

# Main program to choose operation
if __name__ == "__main__":
    image_path = input("Enter the path to your image file: ")
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error loading image. Please check the path.")
    else:
        current_image = original_image.copy()

        while True:
            print("\nChoose an operation to perform:")
            print("1. Colorize Image")
            print("2. Reduce Noise")
            print("3. Inpaint Image (Interactive)")
            print("4. Enhance Sharpness")
            print("5. Remove Scratches and Artifacts (Automatic)")
            print("6. Dehaze Image")
            print("7. Enhance Contrast")
            print("8. Exit")
            choice = input("Enter your choice (1/2/3/4/5/6/7/8): ")

            if choice == '1':
                current_image = colorize_image(original_image)
                cv2.imshow("Colorized Image", current_image)
                cv2.imwrite("colorized_image.jpg", current_image)
                print("Colorized image saved as colorized_image.jpg")

            elif choice == '2':
                current_image = reduce_noise(original_image, method="median")
                cv2.imshow("Denoised Image", current_image)
                cv2.imwrite("denoised_image.jpg", current_image)
                print("Denoised image saved as denoised_image.jpg")

            elif choice == '3':
                current_image = inpaint_image_interactive(original_image)
                cv2.imwrite("interactive_inpainted_image.jpg", current_image)
                print("Interactive inpainted image saved as interactive_inpainted_image.jpg")

            elif choice == '4':
                current_image = enhance_sharpness(original_image)
                cv2.imshow("Enhanced Sharpness", current_image)
                cv2.imwrite("sharpened_image.jpg", current_image)
                print("Sharpened image saved as sharpened_image.jpg")

            elif choice == '5':
                current_image = remove_scratches(original_image)
                cv2.imshow("Scratches Removed", current_image)
                cv2.imwrite("scratches_removed_image.jpg", current_image)
                print("Image with scratches removed saved as scratches_removed_image.jpg")

            elif choice == '6':
                current_image = dehaze_image(original_image)
                cv2.imshow("Dehazed Image", current_image)
                cv2.imwrite("dehazed_image.jpg", current_image)
                print("Dehazed image saved as dehazed_image.jpg")

            elif choice == '7':
                current_image = enhance_contrast(original_image)
                cv2.imshow("Contrast Enhanced Image", current_image)
                cv2.imwrite("contrast_enhanced_image.jpg", current_image)
                print("Contrast enhanced image saved as contrast_enhanced_image.jpg")

            elif choice == '8':
                print("Exiting program.")
                cv2.destroyAllWindows()
                break

            else:
                print("Invalid choice. Please enter a valid option.")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
