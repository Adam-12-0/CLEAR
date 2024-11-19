import os
import cv2
import numpy as np
import piq
import csv
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm
import argparse

# Define metric functions
def calculate_psnr(clean, noisy):
    mse = np.mean((clean - noisy) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

def calculate_ssim(clean, noisy):
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    return ssim(clean_gray, noisy_gray)

def calculate_vif(clean, noisy):
    clean_tensor = transforms.ToTensor()(clean).unsqueeze(0)
    noisy_tensor = transforms.ToTensor()(noisy).unsqueeze(0)
    return piq.vif_p(clean_tensor, noisy_tensor).item()

# Function to evaluate metrics for each noisy video
def evaluate_noisy_video_metrics(clean_video_path, noisy_video_path):
    cap_clean = cv2.VideoCapture(clean_video_path)
    cap_noisy = cv2.VideoCapture(noisy_video_path)

    psnr_values, ssim_values, vif_values = [], [], []

    while True:
        ret_clean, frame_clean = cap_clean.read()
        ret_noisy, frame_noisy = cap_noisy.read()

        if not ret_clean or not ret_noisy:
            break

        psnr_values.append(calculate_psnr(frame_clean, frame_noisy))
        ssim_values.append(calculate_ssim(frame_clean, frame_noisy))
        vif_values.append(calculate_vif(frame_clean, frame_noisy))

    cap_clean.release()
    cap_noisy.release()

    # Calculate averages
    metrics = {
        'Average PSNR': np.mean(psnr_values),
        'Average SSIM': np.mean(ssim_values),
        'Average VIF': np.mean(vif_values),
    }

    return metrics

# Main function to run on noisy videos and save results in a single CSV
def main(test=False):
    clean_dataset_path = "./UCF-101/"
    noisy_datasets = {
        'Gaussian': './output/UCF-101-Gaussian/',
        'SaltPepper': './output/UCF-101-SaltPepper/',
        'Poisson': './output/UCF-101-Poisson/',
        'Speckle': './output/UCF-101-Speckle/',
    }
    output_csv_path = "./metrics/metrics_summary.csv"
    metrics_folder = "./metrics/"

    # Ensure metrics folder is created
    os.makedirs(metrics_folder, exist_ok=True)

    # Check for existing video paths in the CSV to skip already processed videos
    processed_videos = set()
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        with open(output_csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header if it exists
            processed_videos = {row[0] for row in reader}  # Collect paths of already processed videos

    # Open CSV in append mode to continue adding new data
    for noise_type, noisy_dataset_path in noisy_datasets.items():
        categories = [d for d in os.listdir(noisy_dataset_path) if os.path.isdir(os.path.join(noisy_dataset_path, d))]
        total_categories = len(categories)

        # If in test mode, limit to the first category only
        if test:
            categories = categories[:1]

        for i, category in enumerate(categories, start=1):
            # Print progress for each category
            print(f"{noise_type} {i}/{total_categories} - {category}")

            category_path = os.path.join(noisy_dataset_path, category)
            for file in os.listdir(category_path):
                if file.endswith(('.avi', '.mp4')):
                    clean_video_path = os.path.join(clean_dataset_path, category, file)
                    noisy_video_path = os.path.join(category_path, file)

                    # Skip video if already processed
                    if noisy_video_path in processed_videos:
                        # print(f"Skipping already processed video: {noisy_video_path}")
                        continue

                    if os.path.exists(clean_video_path):
                        metrics = evaluate_noisy_video_metrics(clean_video_path, noisy_video_path)

                        # Write each row to the CSV individually in append mode to prevent data loss
                        with open(output_csv_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            if os.path.getsize(output_csv_path) == 0:  # Add header if file is empty
                                writer.writerow(["Video Path", "Noise Type", "Video Category", "Video Name", "Average PSNR", "Average SSIM", "Average VIF"])

                            writer.writerow([
                                noisy_video_path, noise_type, category, file,
                                metrics['Average PSNR'], metrics['Average SSIM'], metrics['Average VIF']
                            ])
                            file.flush()  # Ensure each write is saved immediately

                    else:
                        print(f"Warning: Clean video {file} not found for comparison.")

            # Print completion message for each category
            print(f"Finished Processing {noise_type} {i}/{total_categories} - {category}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate noisy video metrics.")
    parser.add_argument("--test", action="store_true", help="Run the script in test mode (process only one category per noise type)")
    args = parser.parse_args()

    main(test=args.test)
