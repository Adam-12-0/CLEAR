# %%
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import os
import piq # for vif
import torch
from torchvision import transforms
import argparse
import csv

# %%
def denoise_fastnlm(frame, h, hColor, templateWindowSize, searchWindowSize):
    """
    Applies Fast Non-Local Means Denoising to a video frame.

    Parameters:
        frame (numpy.ndarray): The input noisy video frame.
        h (int): Parameter regulating filter strength for luminance.
        hColor (int): Parameter regulating filter strength for color.
        templateWindowSize (int): Size in pixels of the template patch.
        searchWindowSize (int): Size in pixels of the window used to search for patches.

    Returns:
        denoised_frame (numpy.ndarray): The denoised video frame.
    """
    if frame.dtype != 'uint8' or len(frame.shape) != 3 or frame.shape[2] != 3:
        frame = cv2.convertScaleAbs(frame)

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # Apply GPU denoising 
    denoised_frame = cv2.cuda.fastNlMeansDenoisingColored(
        frame,
        None,
        h,
        hColor,
        templateWindowSize,
        searchWindowSize
    )

    return denoised_frame
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
def process_video(input_video_path, output_video_path, **kwargs):
    """Apply selected filter to each frame of the input video and save the result."""
    print(input_video_path)
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    psnr_values, ssim_values, vif_values = [], [], []
    processing_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        start_time = time.time()
        
        # Get parameters 
        h = kwargs.get('h', 10)
        hColor = kwargs.get('hColor', 10)
        templateWindowSize = kwargs.get('templateWindowSize', 7)
        searchWindowSize = kwargs.get('searchWindowSize', 21)

        # Apply the chosen filter
        denoised_frame = denoise_fastnlm(frame, h, hColor, templateWindowSize, searchWindowSize)

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
        #out.write(denoised_frame)
    
    cap.release()
    #out.release()
    
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

    csv_path = "metrics_2.csv"
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file doesn't exist
        if not csv_exists:
            writer.writerow(["Input Video", "Output Video", "Avg PSNR", "Avg SSIM", "Avg VIF", "Avg Processing Time"])
        
        # Write metrics row
        writer.writerow([input_video_path, output_video_path, f"{avg_psnr:.2f}", f"{avg_ssim:.4f}", f"{avg_vif:.4f}", f"{avg_processing_time:.4f}"])

# %%
def process_directory(input_dir, output_dir, **kwargs):
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
                process_video(input_video_path, output_video_path, **kwargs)

# %%
def main():
    parser = argparse.ArgumentParser(description="Video Denoising with Median and Bilateral Filters")

    # General arguments
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing videos.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for saving denoised videos.")
    
    # Optional fast non-Local means arguments
    #RUN1: 10-10-11-25, metrics.csv, /denoisedVidsDefParas/
    #RUN2: 10-10-7-21, mecrics_2.csv, /denoisedVideos-10-10-7-21/
    parser.add_argument("--h", type=int, default=10, help="Regulates the filter strength for luminance.")
    parser.add_argument("--hColor", type=int, default=10, help="Regulates the filter strength for color.")
    parser.add_argument("--templateWindowSize", type=int, default=7, help="Size in pixels of the template patch.") # must be odd
    parser.add_argument("--searchWindowSize", type=int, default=21, help="Size in pixels of the window used to search for patches.") #must be odd
    '''
        h (int): Parameter regulating filter strength for luminance.
        hColor (int): Parameter regulating filter strength for color.
        templateWindowSize (int): Size in pixels of the template patch.
        searchWindowSize (int): Size in pixels of the window used to search for patches.
    '''
    
    args = parser.parse_args()

    # Print arguments for verification
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    
    # Set filter-specific parameters
    filter_params = {}
    filter_params["h"] = args.h
    print("h: ", args.h)
    filter_params["hColor"] = args.hColor
    print("hColor: ", args.hColor)
    filter_params["templateWindowSize"] = args.templateWindowSize
    print("templateWindowSize: ", args.templateWindowSize)
    filter_params["searchWindowSize"] = args.searchWindowSize
    print("searchWindowSize: ", args.searchWindowSize)

    # Process the directory with the specified filter and parameters
    process_directory(args.input_dir, args.output_dir, **filter_params)

if __name__ == "__main__":
    main()