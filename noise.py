import os
import cv2
import numpy as np
from multiprocessing import Pool
import shutil

# Noise Injection Functions
def add_gaussian_noise(frame, mean=0, sigma=50):
    gaussian = np.random.normal(mean, sigma, frame.shape).astype(np.float32)
    noisy_frame = cv2.add(frame.astype(np.float32), gaussian)
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    return noisy_frame

def add_salt_pepper_noise(frame, salt_prob=0.05, pepper_prob=0.05):
    noisy_frame = frame.copy()
    num_salt = np.ceil(salt_prob * frame.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in frame.shape[:2]]
    noisy_frame[coords[0], coords[1], :] = 255
    num_pepper = np.ceil(pepper_prob * frame.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in frame.shape[:2]]
    noisy_frame[coords[0], coords[1], :] = 0
    return noisy_frame

def add_poisson_noise(frame):
    vals = len(np.unique(frame))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(frame * vals) / float(vals)
    noisy_frame = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_frame

def add_speckle_noise(frame, std=0.1):
    noise = np.random.normal(0, std, frame.shape)
    noisy_frame = frame + frame * noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    return noisy_frame

# Function to check if all videos in a category are processed
def is_category_processed(category_path, output_dirs):
    """
    Checks if all videos in the specified category have been processed.

    Parameters:
        category_path (str): Path to the original category directory.
        output_dirs (dict): Dictionary of output directories.

    Returns:
        bool: True if the category is fully processed, False otherwise.
    """
    video_files = [file for file in os.listdir(category_path) if file.endswith(('.avi', '.mp4'))]
    for noise_type, output_dir in output_dirs.items():
        output_category_path = os.path.join(output_dir, os.path.basename(category_path))
        if not os.path.exists(output_category_path):
            return False

        # Check if all video files are present in the output directory
        for video_file in video_files:
            if not os.path.exists(os.path.join(output_category_path, video_file)):
                return False

    return True

# Function to Process Each Video for All Noise Types
def process_video(video_path):
    base_name = os.path.basename(video_path)
    category = os.path.basename(os.path.dirname(video_path))

    output_dirs = {
    "gaussian": f"./output/UCF-101-Gaussian/{category}",
    "salt_pepper": f"./output/UCF-101-SaltPepper/{category}",
    "poisson": f"./output/UCF-101-Poisson/{category}",
    "speckle": f"./output/UCF-101-Speckle/{category}"
    }

    for output_dir in output_dirs.values():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)  # Allow existing directories without raising an error

    # Open the original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a writer for each noise version
    writers = {
        "gaussian": cv2.VideoWriter(os.path.join(output_dirs["gaussian"], base_name), fourcc, fps, (width, height)),
        "salt_pepper": cv2.VideoWriter(os.path.join(output_dirs["salt_pepper"], base_name), fourcc, fps, (width, height)),
        "poisson": cv2.VideoWriter(os.path.join(output_dirs["poisson"], base_name), fourcc, fps, (width, height)),
        "speckle": cv2.VideoWriter(os.path.join(output_dirs["speckle"], base_name), fourcc, fps, (width, height))
    }

    # Read and process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        noisy_frames = {
            "gaussian": add_gaussian_noise(frame),
            "salt_pepper": add_salt_pepper_noise(frame),
            "poisson": add_poisson_noise(frame),
            "speckle": add_speckle_noise(frame)
        }

        # Write the processed frames
        for noise_type, writer in writers.items():
            writer.write(noisy_frames[noise_type])

    # Release resources
    cap.release()
    for writer in writers.values():
        writer.release()

    # print(f"Finished processing video: {video_path}")

# Main Function for Processing Dataset with Progress Tracking and Test Option
def main(test=False):
    input_dataset_path = "./UCF-101/"
    video_paths = []

    # Gather all video file paths and group by action categories
    category_video_paths = {}
    for root, _, files in os.walk(input_dataset_path):
        category = os.path.basename(root)
        if files:
            video_files = [os.path.join(root, file) for file in files if file.endswith(('.avi', '.mp4'))]
            if video_files:
                category_video_paths[category] = video_files

    # Define output directories
    output_dirs = {
        "gaussian": "./output/UCF-101-Gaussian",
        "salt_pepper": "./output/UCF-101-SaltPepper",
        "poisson": "./output/UCF-101-Poisson",
        "speckle": "./output/UCF-101-Speckle"
    }

    total_categories = len(category_video_paths)
    processed_categories = 0

    # Process each action category
    for category, video_paths in category_video_paths.items():
        print(f"Processing category {processed_categories + 1}/{total_categories}: {category}")

        # Check if the category has already been fully processed
        category_path = os.path.join(input_dataset_path, category)
        if is_category_processed(category_path, output_dirs):
            print(f"Category '{category}' is already fully processed. Skipping...")
            processed_categories += 1
            continue

        # Use multiprocessing to speed up processing within each category
        with Pool(processes=4) as pool:
            pool.map(process_video, video_paths)

        processed_categories += 1

        # Exit after processing the first category if test mode is enabled
        if test:
            print("Test mode enabled: Stopping after processing the first category.")
            break

    print("All categories processed successfully." if not test else "Test category processed successfully.")

if __name__ == "__main__":
    import sys
    # Check if test argument is provided
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    main(test=test_mode)