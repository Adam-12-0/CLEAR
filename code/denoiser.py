# %%
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import os
import piq
import torch
from torchvision import transforms
import argparse

# %%
def apply_median_filter(frame, kernel_size):
    """Apply Median Filter to the frame."""
    return cv2.medianBlur(frame, kernel_size)

# %%
def apply_bilateral_filter(frame, d, sigma_color, sigma_space):
    """Apply Bilateral Filter to the frame."""
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space, borderType=cv2.BORDER_DEFAULT)


# %%
def calculate_psnr(clean, denoised):
    """
    Calculates the Peak Signal-to-Noise Ratio between two frames.

    Parameters:
        clean (numpy.ndarray): The original clean video frame.
        denoised (numpy.ndarray): The denoised video frame.

    Returns:
        psnr (float): The PSNR value in decibels.
    """
    mse = np.mean((clean - denoised) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

# %%
def calculate_vif(clean, denoised):
    """
    Calculates the Visual Information Fidelity between two frames.

    Parameters:
        clean (numpy.ndarray): The original clean video frame.
        denoised (numpy.ndarray): The denoised video frame.

    Returns:
        vif_score (float): The VIF value.
    """
    if clean is None or denoised is None:
        print("Error: One or more frames are None.")
        return np.nan
    clean_tensor = transforms.ToTensor()(clean).unsqueeze(0)
    denoised_tensor = transforms.ToTensor()(denoised).unsqueeze(0)
    vif_loss = piq.VIFLoss()
    vif_score = vif_loss(clean_tensor, denoised_tensor).item()
    return vif_score

# %%
def calculate_metrics(original_frame, denoised_frame):
    """Calculate quality metrics between original and denoised frames."""
    '''print(f'original frame size {original_frame.shape}')
    print(f'denoised frame size {denoised_frame.shape}')'''
    psnr_value = psnr(original_frame, denoised_frame, data_range=original_frame.max() - original_frame.min())
    #psnr_implement = calculate_psnr(original_frame, denoised_frame)
    #print(f'psnr value {psnr_value} vs{psnr_implement}')
    clean_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(clean_gray, denoised_gray, full=True)
    vif_value = calculate_vif(original_frame, denoised_frame)

    return psnr_value, ssim_score, vif_value


# %%
def process_video(input_video_path, output_video_path, filter_type, **kwargs):
    """Apply selected filter to each frame of the input video and save the result."""
    print(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    psnr_values, ssim_values, vif_values = [], [], []
    processing_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        start_time = time.time()
        
        # Apply the chosen filter
        if filter_type == 'median':
            kernel_size = kwargs.get('kernel_size', 3)
            denoised_frame = apply_median_filter(frame, kernel_size)
        elif filter_type == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            denoised_frame = apply_bilateral_filter(frame, d, sigma_color, sigma_space)
        
        end_time = time.time()
        '''print(f'original frame size {original_frame.shape}')
        print(f'denoised frame size {denoised_frame.shape}')'''
        # Calculate metrics
        psnr_value, ssim_value, vif_value = calculate_metrics(original_frame, denoised_frame)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        vif_values.append(vif_value)

        processing_times.append(end_time - start_time)
        
        # Write the denoised frame to output video
        out.write(denoised_frame)
    
    cap.release()
    out.release()
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_vif = np.mean(vif_values)
    avg_processing_time = np.mean(processing_times)
    
    print(f"Processed {input_video_path} -> {output_video_path}")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average VIF: {avg_vif:.4f}")
    print(f"Average Processing Time per Frame: {avg_processing_time:.4f} seconds")

# %%
def process_directory(input_dir, output_dir, filter_type, **kwargs):
    """Recursively process all .avi files in input_dir and save the output to output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        # Process each .avi file found
        for file in files:
            if file.endswith(".avi"):
                input_video_path = os.path.join(root, file)
                
                # Preserve the subdirectory structure in the output directory
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                output_video_path = os.path.join(output_subdir, f"denoised_{file}")
                
                # Process each video
                process_video(input_video_path, output_video_path, filter_type, **kwargs)


'''
input_gaussian = '~/workspace/CAP5415/UCF101/noisy_out/UCF-101-Gaussian'
output_gaussian = '~/workspace/CAP5415/UCF101/denoised/UCF-101-Gaussian'

input_poisson = ~/workspace/CAP5415/UCF101/noisy_out/UCF-101-Poisson'
output_poison = '~/workspace/CAP5415/UCF101/denoised/UCF-101-Poisson'

input_saltpepper = '~/workspace/CAP5415/UCF101/noisy_out/UCF-101-SaltPepper'
output_saltpepper = ~/workspace/CAP5415/UCF101/denoised/UCF-101-SaltPepper'

input_Speckle = '~/workspace/CAP5415/UCF101/noisy_out/UCF-101-Speckle'
output_Speckle = ~/workspace/CAP5415/UCF101/denoised/UCF-101-Speckle'

# %%
process_directory(input_gaussian, output_gaussian, filter_type='median', kernel_size=5)
'''
def main():
    parser = argparse.ArgumentParser(description="Video Denoising with Median and Bilateral Filters")

    # General arguments
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing videos.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for saving denoised videos.")
    parser.add_argument("filter_type", type=str, choices=["median", "bilateral"], help="Type of filter to apply ('median' or 'bilateral').")
    
    # Median filter arguments
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for median filter (must be an odd number).")

    # Bilateral filter arguments
    parser.add_argument("--d", type=int, default=9, help="Diameter of each pixel neighborhood in bilateral filter.")
    parser.add_argument("--sigma_color", type=float, default=75, help="Filter sigma in the color space for bilateral filter.")
    parser.add_argument("--sigma_space", type=float, default=75, help="Filter sigma in the coordinate space for bilateral filter.")
    
    args = parser.parse_args()

    # Print arguments for verification
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Filter Type: {args.filter_type}")
    
    # Set filter-specific parameters
    filter_params = {}
    if args.filter_type == "median":
        filter_params["kernel_size"] = args.kernel_size
        print(f"Median Filter Kernel Size: {args.kernel_size}")
    elif args.filter_type == "bilateral":
        filter_params["d"] = args.d
        filter_params["sigma_color"] = args.sigma_color
        filter_params["sigma_space"] = args.sigma_space
        print(f"Bilateral Filter Parameters - d: {args.d}, sigma_color: {args.sigma_color}, sigma_space: {args.sigma_space}")

    # Process the directory with the specified filter and parameters
    process_directory(args.input_dir, args.output_dir, filter_type=args.filter_type, **filter_params)

if __name__ == "__main__":
    main()

