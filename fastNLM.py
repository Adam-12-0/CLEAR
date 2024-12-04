# %%
import cv2
import numpy as np
import time
import os
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
    # Adjust frame as needed
    if frame.dtype != 'uint8' or len(frame.shape) != 3 or frame.shape[2] != 3:
        frame = cv2.convertScaleAbs(frame)

    # Denoise the frame
    denoised_frame = cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h,
        hColor,
        templateWindowSize,
        searchWindowSize
    )

    # Return the denoised frame
    return denoised_frame

# %%
def process_video(input_video_path, output_video_path, **kwargs):
    """Apply selected filter to each frame of the input video and save the result."""
    # Initialize variables, writer and capture
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    processing_times = []
    
    while cap.isOpened():
        # Read in a frame
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

        # Apply the denoiser to the frame
        denoised_frame = denoise_fastnlm(frame, h, hColor, templateWindowSize, searchWindowSize)

        # Stop the time
        end_time = time.time()
        processing_times.append(end_time - start_time)
        
        # Write the denoised frame to output video
        out.write(denoised_frame)
    
    # Release the writer and capture
    cap.release()
    out.release()
    avg_processing_time = np.mean(processing_times)
    # print(f"Average Processing Time per Frame: {avg_processing_time:.4f} seconds")
    

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
    parser = argparse.ArgumentParser(description="Video Denoising with fast non-local means")

    # General arguments
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing videos.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for saving denoised videos.")
    
    # Optional fast non-Local means arguments
    '''
        h (int): Parameter regulating filter strength for luminance.
        hColor (int): Parameter regulating filter strength for color.
        templateWindowSize (int): Size in pixels of the template patch.
        searchWindowSize (int): Size in pixels of the window used to search for patches.
    '''
    parser.add_argument("--h", type=int, default=10, help="Regulates the filter strength for luminance.")
    parser.add_argument("--hColor", type=int, default=10, help="Regulates the filter strength for color.")
    parser.add_argument("--templateWindowSize", type=int, default=7, help="Size in pixels of the template patch.") # Must be odd
    parser.add_argument("--searchWindowSize", type=int, default=21, help="Size in pixels of the window used to search for patches.") #Must be odd
    
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