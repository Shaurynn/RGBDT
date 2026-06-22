import cv2
import sys

# 1. Load the 16-bit image natively (IMREAD_UNCHANGED is critical here)
image_path = "/home/keef07/calibration_dataset/thermal/pair_01.png"
raw_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if raw_16bit is None:
    print("Could not load image. Check the file path.")
    sys.exit()

# 2. Stretch the hidden data across the 0-255 visual spectrum
normalized_8u = cv2.normalize(raw_16bit, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 3. Apply the CLAHE enhancement to make the checkerboard pop
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
enhanced = clahe.apply(normalized_8u)

# 4. Display the hidden data
cv2.imshow("Original 16-Bit (What the monitor sees)", raw_16bit) # This will look grey/black
cv2.imshow("Contrast Stretched (What the data actually is)", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()