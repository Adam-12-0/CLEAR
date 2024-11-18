import optix
import os
import cv2
import cupy as cp
import cupy.cuda.runtime as cuda
import numpy as np
import argparse
import time
import csv
import re
from tqdm import tqdm

def create_optix_image(width, height, channels):
    # Initialize an OptiX 2D image
    optix_image = optix.Image2D()
    
    # Set image dimensions and format
    optix_image.width = width
    optix_image.height = height
    match channels:
        case 1:
            optix_image.format = optix.PIXEL_FORMAT_FLOAT1
        case 2:
            optix_image.format = optix.PIXEL_FORMAT_FLOAT2
        case 3:
            optix_image.format = optix.PIXEL_FORMAT_FLOAT3
        case _:
            optix_image.format = optix.PIXEL_FORMAT_FLOAT4
    optix_image.pixelStrideInBytes = channels * 4  # 4 channels, 4 bytes per channel
    optix_image.rowStrideInBytes = optix_image.width * optix_image.pixelStrideInBytes

    # Allocate device memory for the image data
    optix_image.data = cuda.malloc(optix_image.height * optix_image.rowStrideInBytes)

    return optix_image

def free_optix_image(optix_image):
    # Free the device memory allocated for the OptiX image
    cuda.free(optix_image.data)
    
    # Set the image data pointer to null (0)
    optix_image.data = 0

def copy_cpu_to_gpu(image_data, optix_image):
    # Copy data from host to device   
    cuda.memcpy(
        optix_image.data,
        image_data.ctypes.data,
        optix_image.rowStrideInBytes * optix_image.height,
        cuda.memcpyHostToDevice
    )

def copy_gpu_to_cpu(optix_image, image_data):
    # Copy data from device to host
    cuda.memcpy(
        image_data.ctypes.data,
        optix_image.data,
        optix_image.height * optix_image.rowStrideInBytes,
        cuda.memcpyDeviceToHost
    )

def create_denoiser(context, denoiser_layer, tile_size, flow):
    # Initialize denoiser options
    options = optix.DenoiserOptions()
    options.guideAlbedo = 0
    options.guideNormal = 0
    
    # Create the denoiser (assume LDR for 0-255 color range)
    if flow > 0:
        denoiser = context.denoiserCreate(optix.DENOISER_MODEL_KIND_TEMPORAL, options)
    else:
    	denoiser = context.denoiserCreate(optix.DENOISER_MODEL_KIND_LDR, options)

    # Get recommended sizes for memory resources
    sizes = denoiser.computeMemoryResources(tile_size[0], tile_size[1])

    image_width = denoiser_layer.input.width
    image_height = denoiser_layer.input.height

    scratch_size = 0
    overlap = 0
    if tile_size[0] == image_width and tile_size[1] == image_height:
        scratch_size = sizes.withoutOverlapScratchSizeInBytes
    else:
        scratch_size = sizes.withOverlapScratchSizeInBytes
        overlap = sizes.overlapWindowSizeInPixels

    # Allocate device memory for state and scratch space
    d_state = cp.empty((sizes.stateSizeInBytes), dtype='B')
    d_scratch = cp.empty((scratch_size), dtype='B')

    # Setup the denoiser
    denoiser.setup(
        0,
        tile_size[0] + 2 * overlap,
        tile_size[1] + 2 * overlap,
        d_state.data.ptr,
        d_state.nbytes,
        d_scratch.data.ptr,
        d_scratch.nbytes
    )

    return denoiser, d_state, d_scratch, overlap

def run_denoiser(denoiser, denoiser_layer, guide_layer, d_state, d_scratch, tile_size, overlap):
    # Set up denoiser parameters
    params = optix.DenoiserParams()
    params.blendFactor = 0.0

    # Invoke the denoiser
    denoiser.invokeTiled(
        0,
        params,
        d_state.data.ptr,
        d_state.nbytes,
        guide_layer,
        [denoiser_layer],
        d_scratch.data.ptr,
        d_scratch.nbytes,
        overlap,
        tile_size[0],
        tile_size[1]
    )

def process_video(video_in, video_out, args):
    # Start the timer
    start_time = time.time()

    # Open the noisy video
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Open the flow video (if requested)
    cap_flow = None
    if args.flow == 2:
        cap_flow = cv2.VideoCapture(re.sub(rf'{re.escape(args.input)}[^/]+', args.flowdir, video_in))
    
    # Ensure output directory exists
    dir_out = os.path.dirname(video_out)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out, exist_ok=True)  # Allow existing directories without raising an error

    # Create a writer for denoised output video
    writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    # Allocate CPU memory to hold noisy and denoised frame data in RGBA_F32 format
    frame_rgba_f32_in = np.zeros((height, width, 4), dtype=np.float32)
    frame_rgba_f32_out = np.zeros((width * height * 4), dtype=np.float32)
    
    # Allocate CPU memory to hold denoised frame data in RGB_UINT8 format
    frame_bgr_u8_out = np.zeros((height, width, 3), dtype=np.uint8)

    # Initialize the denoiser layer and GPU memory
    denoiser_layer = optix.DenoiserLayer()
    denoiser_layer.input = create_optix_image(width, height, 4)
    optix_outputs = [create_optix_image(width, height, 4), create_optix_image(width, height, 4)]
    denoiser_layer.output = optix_outputs[0]

    # Initialize guide layer for 2d flow
    guide_layer = optix.DenoiserGuideLayer()
    if args.flow > 0:
    	guide_layer.flow = create_optix_image(width, height, 2)

    # Determine the tile size
    tile_size = (width, height) if args.tilesize[0] <= 0 or args.tilesize[1] <= 0 else args.tilesize

    # Create device context and denoiser
    context = optix.deviceContextCreate(0, optix.DeviceContextOptions())
    denoiser, d_state, d_scratch, overlap = create_denoiser(context, denoiser_layer, tile_size, args.flow)

    # Initialize variables for optical flow
    prev_frame_gray = None

    # Read and process each frame
    frame_count = 0
    frame_times = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start the frame timer
        start_frame_time = time.time()

	# Convert frame to grayscale for optical flow calculation
        frame_gray = None
        if args.flow == 1:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame_gray is not None:
                # Calculate optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Copy the 2d flow data from CPU to GPU
                copy_cpu_to_gpu(flow, guide_layer.flow)
        elif args.flow == 2:
            ret_flow, frame_flow = cap_flow.read()
            
            if ret_flow:
                # Convert the BGR image back to HSV
                hsv = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2HSV)
                hue = hsv[..., 0]  # Hue represents the angle/direction of the flow
                value = hsv[..., 2]  # Value represents the magnitude of the flow

                # Normalize the Value to range [0, 255]
                magnitude = cv2.normalize(value.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
                
                # Convert the Hue to radians (was scaled to [0, 180] in HSV, convert back to [0, 360 degrees])
                angle = hue * (np.pi / 180) * 2
                
                # Reconstruct the optical flow vectors
                flow_x = magnitude * np.cos(angle)
                flow_y = magnitude * np.sin(angle)
                
                # Copy the 2d flow data from CPU to GPU
                copy_cpu_to_gpu(np.stack((flow_x, flow_y), axis=-1), guide_layer.flow)

        # Split the noisy BGR frame into its red, green, and blue channels using slicing
        b, g, r = cv2.split(frame)

        # Initialize our array to hold the frame data in F32 format with an alpha channel
        frame_rgba_f32_in[:, :, 0] = np.array(r, dtype=np.float32) / 255.0  # Normalize red channel
        frame_rgba_f32_in[:, :, 1] = np.array(g, dtype=np.float32) / 255.0  # Normalize green channel
        frame_rgba_f32_in[:, :, 2] = np.array(b, dtype=np.float32) / 255.0  # Normalize blue channel
        frame_rgba_f32_in[:, :, 3] = 1.0  # Set alpha channel to 1 (opaque)

        # Copy the frame data from CPU to GPU
        copy_cpu_to_gpu(frame_rgba_f32_in, denoiser_layer.input)

        # Run the denoiser
        run_denoiser(denoiser, denoiser_layer, guide_layer, d_state, d_scratch, tile_size, overlap)

        # Transfer the denoised image data from GPU to CPU
        copy_gpu_to_cpu(denoiser_layer.output, frame_rgba_f32_out)

        # Extract and normalize RGB channels from the data
        frame_bgr_u8_out[..., 2] = (frame_rgba_f32_out[0::4].reshape(height, width) * 255).clip(0, 255).astype(np.uint8)  # Red channel
        frame_bgr_u8_out[..., 1] = (frame_rgba_f32_out[1::4].reshape(height, width) * 255).clip(0, 255).astype(np.uint8)  # Green channel
        frame_bgr_u8_out[..., 0] = (frame_rgba_f32_out[2::4].reshape(height, width) * 255).clip(0, 255).astype(np.uint8)  # Blue channel

        # Write the processed frame
        writer.write(frame_bgr_u8_out)

        # Ping pong output GPU buffers
        if args.flow > 0:
            temp = denoiser_layer.previousOutput if frame_count > 0 else optix_outputs[1]
            denoiser_layer.previousOutput = denoiser_layer.output
            denoiser_layer.output = temp
            prev_frame_gray = frame_gray
            
        # Add to total frame time and update counter
        frame_times += (time.time() - start_frame_time) * 1000
        frame_count += 1

    # Release resources
    cap.release()
    writer.release()
    free_optix_image(denoiser_layer.input)
    free_optix_image(optix_outputs[0])
    free_optix_image(optix_outputs[1])
    if args.flow:
    	free_optix_image(guide_layer.flow)
    context.destroy()
    denoiser.destroy()
    
    # Return times
    elapsed_time = (time.time() - start_time) * 1000
    average_frame_time = frame_times / frame_count
    return elapsed_time, average_frame_time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Denoise the specified video dataset using NVIDIA OptiX'
    )
    
    parser.add_argument(
        '-i', '--input',
        metavar='dir',
        type=str,
        help="Input dataset directory (default: './output/')",
        default='./output/'
    )
    
    parser.add_argument(
        '-o', '--output',
        metavar='dir',
        type=str,
        help="Output dataset directory (default: './output_optix/')",
        default='./output_optix/'
    )
    
    parser.add_argument(
        '-d', '--flowdir',
        metavar='dir',
        type=str,
        help="Input dataset directory (default: './UCF-101_flow/')",
        default='./UCF-101_flow/'
    )

    parser.add_argument(
        '-t', '--tilesize',
        metavar='INT',
        type=int,
        nargs=2,
        help="Specify the tile size for processing.",
        default=(0, 0)
    )

    parser.add_argument(
        '-p', '--perf',
        metavar='dir',
        type=str,
        help="File where performance data will be dumped per video (default: 'performance.csv')",
        default='performance.csv'
    )

    parser.add_argument(
    	'-f', '--flow',
        metavar='INT',
        type=int,
    	help='Enable 2d flow guide layer (0 - disable, 1 - generate flow from noisy videos, 2 - use flow from UCF-101_flow).',
        default=0
    )
    
    parser.add_argument(
    	'-l', '--limit',
        metavar='INT',
        type=int,
    	help='Image count limit per category and per noise type (0 for no limit).',
        default=0
    )

    args = parser.parse_args()

    print("\n")
    print("Arguments:")
    print("    input:\t", args.input)
    print("    output:\t", args.output)
    print("    tilesize:\t", args.tilesize)
    print("    perf:\t", args.perf)
    print("    flow:\t", args.flow)
    print("    flowdir:\t", args.flowdir)
    print("\n")
    
    return args

def main():
    args = parse_args()

    skip_videos = []
    if os.path.exists(args.perf):
        with open(args.perf, mode='r', newline='') as file:
            reader = csv.reader(file)
            # Assuming video path is in the first column (index 0)
            skip_videos = [row[0] for row in reader]

    path_dict = {}
    skip_count = 0
    video_paths = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.endswith(('.avi', '.mp4')):
                file_path = os.path.join(root, file)
                video_count = path_dict.get(root, 0) + 1
                path_dict[root] = video_count
                if video_count < args.limit or args.limit <= 0:
                    if file_path not in skip_videos:
                        video_paths.append(file_path)
                    else:
                        skip_count += 1

    if skip_count > 0:
        print(f"Skipping {skip_count} videos due to being processed in a previous run...\n")

    with open(args.perf, mode='a', newline='') as file:
    	writer = csv.writer(file)
    	writer.writerow(["Video Path", "Elapsed Time (ms)", "Average Frame Time (ms)"])
    	for video_path in tqdm(video_paths):
    	    elapsed_time, average_frame_time = process_video(video_path, video_path.replace(args.input, args.output, 1), args)
    	    writer.writerow([video_path, elapsed_time, average_frame_time])

if __name__ == "__main__":
    main()

