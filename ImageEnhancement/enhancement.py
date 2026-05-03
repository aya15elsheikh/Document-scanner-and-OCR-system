import cv2
import numpy as np

def enhance_image(image: np.ndarray) -> np.ndarray:
    # 1. Grayscale only
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Light denoise ONLY if needed
    noise_est = cv2.Laplacian(gray, cv2.CV_64F).var()
    if noise_est < 30:  # only very noisy images
        gray = cv2.fastNlMeansDenoising(gray, None, h=6)

    # 3. No resizing unless extremely small (prevents blur)
    h, w = gray.shape
    if max(h, w) < 600:
        gray = cv2.resize(gray, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

    # 4. Mild contrast only
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Adaptive threshold directly (no heavy preprocessing chain)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    # 6. Polarity fix
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary
    