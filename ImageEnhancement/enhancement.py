import cv2
import numpy as np

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Applies Image Enhancement techniques (Bilateral Filtering and Adaptive Thresholding)
    to eliminate shadows and improve text-to-background contrast.

    Args:
        image (np.ndarray): The input image (can be BGR or grayscale).

    Returns:
        np.ndarray: The enhanced, binarized image.
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Bilateral Filtering to reduce noise while preserving edges
    # Parameters: d (diameter of pixel neighborhood), sigmaColor, sigmaSpace
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply Adaptive Thresholding
    # Parameters: src, maxValue, adaptiveMethod, thresholdType, blockSize, C
    enhanced = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return enhanced
