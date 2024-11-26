# main_test_vrt.py

import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader
import csv
import time

# Removed imports related to metrics
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim
# from sewar.full_ref import vifp  # For VIF computation

from models.network_vrt import VRT as net
from utils import utils_image as util

# Removed the compute_vif function
# def compute_vif(original_frame, denoised_frame):
#     vif_value = vifp(original_frame, denoised_frame)
#     return vif_value

def main():
    # Define noise types and sigma values
    # noise_types = ['Gaussian', 'Poisson', 'SaltPepper', 'Speckle']
    noise_types = [ 'Poisson', 'SaltPepper', 'Speckle']
    noise_sigma_mapping = {
        # 'Gaussian': 10,
        'Poisson': 10,
        'SaltPepper': 30,
        'Speckle': 20,
    }

    # Base directories relative to the script location
    base_input_dir = '../UCF101/output/'
    base_output_dir = '../UCF101/denoised/'
    original_dir = '../UCF101/UCF-101/'

    # Create an args object with necessary attributes
    class Args:
        pass

    args = Args()
    args.task = '008_VRT_videodenoising_DAVIS'  # Using the denoising task
    args.scale = 1
    args.window_size = [6, 8, 8]
    args.tile = [4, 128, 128]   # Adjust as necessary
    args.tile_overlap = [2, 20, 20]
    args.num_workers = 4
    args.nonblind_denoising = True  # Since we have known sigma
    args.save_result = False  # We'll handle saving ourselves
    args.folder_lq = None
    args.folder_gt = None

    # Prepare the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)

    # Iterate over noise types
    for noise in noise_types:
        sigma_value = noise_sigma_mapping.get(noise, 10)
        input_dir = os.path.join(base_input_dir, f'UCF-101-{noise}')
        output_dir = os.path.join(base_output_dir, f'UCF-101-{noise}_denoise')
        metrics_file = os.path.join(output_dir, f'metrics_{noise}.csv')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f'\nProcessing noise type: {noise}')
        print(f'Input directory: {input_dir}')
        print(f'Output directory: {output_dir}')
        print(f'Original directory: {original_dir}')

        # Initialize CSV file for metrics
        with open(metrics_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Updated CSV header
            csvwriter.writerow(['Video', 'Processing Time per Frame (s)', 'FPS'])

            # Traverse the directory structure
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.avi'):
                        input_video_path = os.path.join(root, file)
                        # Compute relative path to maintain directory structure
                        relative_path = os.path.relpath(root, input_dir)
                        # Corresponding output directory
                        output_video_dir = os.path.join(output_dir, relative_path)
                        if not os.path.exists(output_video_dir):
                            os.makedirs(output_video_dir)
                        output_video_path = os.path.join(output_video_dir, file)

                        # Added check to skip if output video already exists
                        if os.path.exists(output_video_path):
                            print(f'Skipping {input_video_path} as output already exists.')
                            continue

                        # Corresponding original video path
                        original_video_path = os.path.join(original_dir, relative_path, file)

                        # Check if original video exists
                        if not os.path.exists(original_video_path):
                            print(f'Original video not found: {original_video_path}')
                            continue

                        # Process the video
                        process_video(input_video_path, original_video_path, output_video_path, csvwriter, sigma_value, args, model, device)


def process_video(input_video_path, original_video_path, output_video_path, csvwriter, sigma_value, args, model, device):
    # Read the input and original video frames
    cap_input = cv2.VideoCapture(input_video_path)
    cap_original = cv2.VideoCapture(original_video_path)

    frames_input = []
    frames_original = []

    while True:
        ret_input, frame_input = cap_input.read()
        ret_original, frame_original = cap_original.read()
        if not ret_input or not ret_original:
            break
        frames_input.append(frame_input)
        frames_original.append(frame_original)

    cap_input.release()
    cap_original.release()

    num_frames = len(frames_input)

    if num_frames == 0:
        print(f'No frames found in {input_video_path}')
        return

    # Prepare lq and gt tensors
    lq = np.array(frames_input).astype(np.float32) / 255.0  # Shape (T, H, W, C)
    gt = np.array(frames_original).astype(np.float32) / 255.0

    # Rearrange dimensions to (B, D, C, H, W)
    lq = torch.from_numpy(lq).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, D, C, H, W)
    gt = torch.from_numpy(gt).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    # Add sigma map for non-blind denoising
    if args.nonblind_denoising:
        sigma_map = torch.ones((lq.size(0), lq.size(1), 1, lq.size(3), lq.size(4))).to(device) * (sigma_value / 255.0)
        lq = torch.cat([lq, sigma_map], dim=2)

    # Start timer
    start_time = time.time()

    # Process the video using the model
    with torch.no_grad():
        output = test_video(lq, model, args)

    # End timer
    end_time = time.time()
    total_processing_time = end_time - start_time
    processing_time_per_frame = total_processing_time / num_frames

    # Convert output to numpy array
    output = output.squeeze(0).cpu().numpy()  # Shape (D, C, H, W)

    # Prepare output video writer
    frame_height, frame_width = frames_input[0].shape[:2]
    fps = cap_input.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Default FPS if not available

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Write frames to output video
    for i in range(num_frames):
        denoised_frame = output[i]
        denoised_frame = np.clip(denoised_frame, 0, 1)
        denoised_frame = denoised_frame.transpose(1, 2, 0)  # Convert to HWC

        denoised_frame_uint8 = (denoised_frame * 255.0).round().astype(np.uint8)
        out.write(denoised_frame_uint8)

    out.release()

    # Write metrics to CSV
    csvwriter.writerow([os.path.basename(input_video_path), processing_time_per_frame, fps])

    # Print status
    print(f'Processed {os.path.basename(input_video_path)}: {num_frames} frames, Total Time: {total_processing_time:.2f}s, FPS: {fps}')


def prepare_model_dataset(args):
    ''' Prepare model according to args.task. '''
    # Define model based on the task
    if args.task == '008_VRT_videodenoising_DAVIS':
        model = net(
            upscale=1,
            img_size=[6, 192, 192],
            window_size=[6, 8, 8],
            depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
            indep_reconsts=[9, 10],
            embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            pa_frames=2,
            deformable_groups=16,
            nonblind_denoising=True
        )
        args.scale = 1
        args.window_size = [6, 8, 8]
        args.nonblind_denoising = True
    else:
        raise ValueError(f'Unknown task: {args.task}')

    # Download model if not available
    model_path = f'model_zoo/vrt/{args.task}.pth'
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = f'https://github.com/JingyunLiang/VRT/releases/download/v0.0/{os.path.basename(model_path)}'
        r = requests.get(url, allow_redirects=True)
        print(f'Downloading model {model_path}')
        open(model_path, 'wb').write(r.content)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def test_video(lq, model, args):
    '''Test the video as a whole or in clips (divided temporally).'''
    num_frame_testing = args.tile[0]
    if num_frame_testing:
        # Test as multiple clips if out-of-memory
        sf = args.scale
        num_frame_overlapping = args.tile_overlap[0]
        not_overlap_border = False
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros(b, d, 1, 1, 1)

        for d_idx in d_idx_list:
            lq_clip = lq[:, d_idx:d_idx + num_frame_testing, ...]
            out_clip = test_clip(lq_clip, model, args)
            out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlapping // 2:, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2:, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, :num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, :num_frame_overlapping // 2, ...] *= 0

            E[:, d_idx:d_idx + num_frame_testing, ...].add_(out_clip)
            W[:, d_idx:d_idx + num_frame_testing, ...].add_(out_clip_mask)
        output = E.div_(W)
    else:
        # Test as one clip (the whole video) if you have enough memory
        window_size = args.window_size
        d_old = lq.size(1)
        d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
        output = test_clip(lq, model, args)
        output = output[:, :d_old, :, :, :]

    return output


def test_clip(lq, model, args):
    '''Test the clip as a whole or as patches.'''
    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'Testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # Divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # Test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
        w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size // 2:, :] *= 0
                        out_patch_mask[..., -overlap_size // 2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size // 2:] *= 0
                        out_patch_mask[..., :, -overlap_size // 2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size // 2, :] *= 0
                        out_patch_mask[..., :overlap_size // 2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size // 2] *= 0
                        out_patch_mask[..., :, :overlap_size // 2] *= 0

                E[..., h_idx * sf:(h_idx + size_patch_testing) * sf, w_idx * sf:(w_idx + size_patch_testing) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + size_patch_testing) * sf, w_idx * sf:(w_idx + size_patch_testing) * sf].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old * sf, :w_old * sf]

    return output


if __name__ == '__main__':
    main()