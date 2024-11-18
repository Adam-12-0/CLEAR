import os
import cv2
import numpy as np
import argparse
import piq
import re
import csv
import shutil
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import concurrent.futures
import threading
from torchvision import transforms

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

def calculate_ssim(clean, denoised):
    """
    Calculates the Structural Similarity Index Measure between two frames.

    Parameters:
        clean (numpy.ndarray): The original clean video frame.
        denoised (numpy.ndarray): The denoised video frame.

    Returns:
        ssim_score (float): The SSIM value.
    """
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(clean_gray, denoised_gray, full=True)
    return ssim_score

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

def evaluate_denoising_metrics(denoised_video_path, avg_frame_time, args):
    if not os.path.exists(denoised_video_path):
        print(f"Error: Denoised video not found at {denoised_video_path}")
        return None

    path_parts = denoised_video_path.split('/')
    noise_type = path_parts[2].split('-')[-1] # e.g. "SaltPepper"
    category = path_parts[3]                     # e.g. "PizzaTossing"
    filename = path_parts[4]                     # e.g. "v_PizzaTossing_g01_c01.avi"
    original_video_path = args.clean + category + "/" + filename

    if not os.path.exists(original_video_path):
        print(f"Error: Original video not found at {original_video_path}")
        return None

    cap_original = cv2.VideoCapture(original_video_path)
    cap_noisy = cv2.VideoCapture(denoised_video_path)

    if not cap_original.isOpened() or not cap_noisy.isOpened():
        print("Error: Could not open video file(s).")
        return None

    psnr_values = []
    ssim_values = []
    vif_values = []

    while True:
        ret_orig, frame_orig = cap_original.read()
        ret_noisy, frame_noisy = cap_noisy.read()

        if not ret_orig or not ret_noisy:
            break

        if frame_orig is None or frame_noisy is None:
            print("Warning: Encountered a None frame, skipping.")
            continue

        # Calculate PSNR, SSIM, and VIF
        psnr_values.append(calculate_psnr(frame_orig, frame_noisy))
        ssim_values.append(calculate_ssim(frame_orig, frame_noisy))
        vif_values.append(calculate_vif(frame_orig, frame_noisy))

    cap_original.release()
    cap_noisy.release()

    if len(psnr_values) == 0 or len(ssim_values) == 0 or len(vif_values) == 0:
        print("Error: No valid frames were processed.")
        return None

    # Calculate average metrics
    metrics = {
        'Average PSNR': np.nanmean(psnr_values),
        'Average SSIM': np.nanmean(ssim_values),
        'Average VIF': np.nanmean(vif_values),
        'Average Processing Time (ms)': avg_frame_time,
        'Frames Per Second (FPS)': 1000 / avg_frame_time if avg_frame_time > 0 else 0
    }

    #for key, value in metrics.items():
    #    print(f"{key}: {value:.2f}")

    return noise_type, category, filename, metrics

def worker_thread(video_path, avg_frame_time, args, noise_type_metrics, writer, lock):
    # Evaluate denoising metrics
    noise_type, category, filename, metrics = evaluate_denoising_metrics(video_path, avg_frame_time, args)

    # Serialize the writing to CSV and  noise_type_metrics
    with lock:
        # Write to the CSV
        writer.writerow([
            noise_type, category, filename,
            metrics['Average PSNR'], metrics['Average SSIM'],
            metrics['Average VIF'], metrics['Average Processing Time (ms)'],
            metrics['Frames Per Second (FPS)']
        ])

        # Init noise_type_metrics if not present
        if noise_type not in noise_type_metrics:
            noise_type_metrics[noise_type] = {
                "psnr_sum": 0, "ssim_sum": 0, "vif_sum": 0,
                "time_sum": 0, "fps_sum": 0, "count": 0
            }

        # Update noise_type_metrics
        noise_type_metrics[noise_type]["psnr_sum"] += metrics['Average PSNR']
        noise_type_metrics[noise_type]["ssim_sum"] += metrics['Average SSIM']
        noise_type_metrics[noise_type]["vif_sum"] += metrics['Average VIF']
        noise_type_metrics[noise_type]["time_sum"] += metrics['Average Processing Time (ms)']
        noise_type_metrics[noise_type]["fps_sum"] += metrics['Frames Per Second (FPS)']
        noise_type_metrics[noise_type]["count"] += 1

def parse_args():
    parser = argparse.ArgumentParser(
        description='Denoise the specified video dataset using NVIDIA OptiX'
    )
    
    parser.add_argument(
        '-c', '--clean',
        metavar='dir',
        type=str,
        help="Input clean dataset directory (default: './UCF-101/')",
        default='./UCF-101/'
    )

    parser.add_argument(
        '-d', '--denoised',
        metavar='dir',
        type=str,
        help="Input denoised dataset directory (default: './output_optix/')",
        default='./output_optix/'
    )

    parser.add_argument(
        '-p', '--perf',
        metavar='dir',
        type=str,
        help="Input file for performance data per video (default: 'performance.csv')",
        default='performance.csv'
    )
    
    parser.add_argument(
        '-o', '--output',
        metavar='dir',
        type=str,
        help="Output file for metrics per video (default: 'metrics.csv')",
        default='metrics.csv'
    )

    args = parser.parse_args()

    print("\n")
    print("Arguments:")
    print("    clean:\t", args.clean)
    print("    denoised:\t", args.denoised)
    print("    perf:\t", args.perf)
    print("    output:\t", args.output)
    print("\n")
    
    return args


# Main Function for Processing Dataset with Progress Tracking and Test Option
def main():

    args = parse_args()

    # Parse perf file
    video_to_avg_frame_time = {}
    perf_header = ['Video Path', 'Elapsed Time (ms)', 'Average Frame Time (ms)']
    with open(args.perf, mode='r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            if row == perf_header:
                # Skip this row since it's a header (can be multiple in file if denoised in parts)
                continue

            # Process row
            video_to_avg_frame_time[row[0]] = float(row[2])
    
    # Accumulate metrics per noise_type
    noise_type_metrics = {}
    lock = threading.Lock()

    with open(args.output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Noise Type", "Video Category", "Video Name", "Average PSNR", "Average SSIM", "Average VIF", "Average Processing Time (ms)", "Frames Per Second (FPS)"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []

            # Initialize progress bar
            progress_bar = tqdm(total=len(video_to_avg_frame_time), desc="Processing videos", unit="video")

            for video_path, avg_frame_time in video_to_avg_frame_time.items():
                # Submit each task to the thread pool
               	denoised_path = video_path.replace("./output/", args.denoised)
                futures.append(executor.submit(worker_thread, denoised_path, avg_frame_time, args, noise_type_metrics, writer, lock))

            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()
                progress_bar.update(1)
                
            # Afterwards close the progress bar
            progress_bar.close()

    for noise_type, values in noise_type_metrics.items():
        video_count = values["count"]
        avg_psnr = values["psnr_sum"] / video_count
        avg_ssim = values["ssim_sum"] / video_count
        avg_vif = values["vif_sum"] / video_count
        avg_time = values["time_sum"] / video_count
        avg_fps = values["fps_sum"] / video_count
        
        print (noise_type)
        print ("\tVideo Count: " + str(video_count))
        print ("\tAverage PSNR: " + str(avg_psnr))
        print ("\tAverage SSIM: " + str(avg_ssim))
        print ("\tAverage VIF: " + str(avg_vif))
        print ("\tAverage Frame Time: " + str(avg_time))
        print ("\tAverage FPS: " + str(avg_fps))
        print ("\n")

    print("Done!")

if __name__ == "__main__":
    main()
