import cv2
import numpy as np

def preprocess_frame(frame):
    """
    4-Step Advanced Edge-AI Preprocessing Pipeline for Border Surveillance:
    1. Fast Non-Local Means Denoising (Removes digital grain & night-vision noise)
    2. Atmospheric Normalization (Cuts through dust, morning mist, and heat haze)
    3. CLAHE Low-Light Enhancement (Boosts shadow details without blowing highlights)
    4. Unsharp Masking (Sharpens edges for tiny, distant targets like drones/weapons)
    """
    
    # --- Step 1: Noise & Grain Reduction ---
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
    
    # --- Step 2: Atmospheric & Weather Filtering (Haze/Dust Correction) ---
    norm_frame = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # --- Step 3: Low-Light & Night Enhancement (CLAHE in LAB color space) ---
    lab = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    bright_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # --- Step 4: Edge Sharpening for Distance ---
    gaussian = cv2.GaussianBlur(bright_frame, (0, 0), sigmaX=2.0)
    sharpened_frame = cv2.addWeighted(bright_frame, 1.5, gaussian, -0.5, 0)
    
    return sharpened_frame