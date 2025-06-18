# Nur Aisyah_F55123042
#Project 2 : Enhanced Night vision (using basics libraries only)
#teknik pengolahan citra yang digunakan yaitu: contrast stretching atau improvement, gamma correction, adaptive brightness boost, dan median filter untuk mengurangi noise.
#semua teknik ini diimplementasikan dengan manual  dan hanya menggunakan bantuan numpy
#openCV hanya digunakan untuk membaca dan menampilkan gambar

import cv2
import numpy as np

# --- Load Image ---
img = cv2.imread(r"C:\Users\user\Documents\Downloads\Street lights.jpeg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan! Cek path dan nama file.")
img = img.astype(np.uint8)

# --- Contrast Stretching ---
def contrast_stretching_rgb(image):
    out = np.zeros_like(image)
    for c in range(3):
        ch = image[:, :, c]
        min_val = np.min(ch)
        max_val = np.max(ch)
        out[:, :, c] = ((ch - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
    return out

# --- Gamma Correction ---
def gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv_gamma) * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    return table[image]

# --- Adaptive Brightness Boost ---
def adaptive_brightness(image, factor=1.8):
    boosted = image.astype(np.float32) * factor
    boosted = np.clip(boosted, 0, 255).astype(np.uint8)
    return boosted

# --- Median Filter (Noise Minimization) ---
def median_filter_rgb(image, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0,0)), mode='edge')
    filtered = np.zeros_like(image)
    for c in range(3):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                filtered[i,j,c] = np.uint8(np.median(region))
    return filtered

# --- Processing Pipeline ---
step1 = contrast_stretching_rgb(img)
step2 = gamma_correction(step1, gamma=1.2)
step3 = adaptive_brightness(step2, factor=1.2)
final_output = median_filter_rgb(step3, kernel_size=3)

# --- Display ---
cv2.imshow("Original", img)
cv2.imshow("Contrast Stretched", step1)
cv2.imshow("Gamma Corrected", step2)
cv2.imshow("Brightness Boosted", step3)
cv2.imshow("Final Enhanced (Median Filtered)", final_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
