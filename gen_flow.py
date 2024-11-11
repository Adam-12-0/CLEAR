import os
import cv2
import numpy as np
from multiprocessing import Pool
import shutil

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
def process_video(input_path):
    input_dataset_path = "./UCF-101/"
    output_dataset_path = "./UCF-101_flow/"
    output_path = input_path.replace(input_dataset_path, output_dataset_path, 1)

    # Ensure output directory exists
    dir_out = os.path.dirname(output_path)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out, exist_ok=True)  # Allow existing directories without raising an error

    # Open the original video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a writer for denoised flow video output
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and process each frame
    prev_frame_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

	# Convert frame to grayscale for optical flow calculation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bgr_flow = np.zeros((height, width, 3), dtype=np.uint8)
        if prev_frame_gray is not None:
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Convert flow to polar coordinates (magnitude and angle)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Normalize magnitude to the range [0, 255] for visualization
            mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Convert angle to hue in HSV space (angle corresponds to the hue in [0, 360])
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue (0-180)
            hsv[..., 1] = 255  # Saturation (max)
            hsv[..., 2] = mag_normalized.astype(np.uint8)  # Value (brightness)

            # Convert HSV to BGR (OpenCV uses BGR format by default)
            bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Write the processed frame
        writer.write(bgr_flow)
        
        # Prep for next iteration
        prev_frame_gray = frame_gray

    # Release resources
    cap.release()
    writer.release()

# Main Function for Processing Dataset with Progress Tracking and Test Option
def main():
    input_dataset_path = "./UCF-101/"

    # Gather all video file paths and group by action categories
    category_video_paths = {}
    for root, _, files in os.walk(input_dataset_path):
        category = os.path.basename(root)
        if files:
            video_files = [os.path.join(root, file) for file in files if file.endswith(('.avi', '.mp4'))]
            if video_files:
                category_video_paths[category] = video_files

    total_categories = len(category_video_paths)
    processed_categories = 0

    # Process each action category
    for category, video_paths in category_video_paths.items():
        print(f"Processing category {processed_categories + 1}/{total_categories}: {category}")

        # Use multiprocessing to speed up processing within each category
        with Pool(processes=16) as pool:
            pool.map(process_video, video_paths)

        processed_categories += 1

    print("All categories processed successfully.")

if __name__ == "__main__":
    main()
