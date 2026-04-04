from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers import AutoencoderDC
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DC-AE Tokenization Script")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing input fMRI frames.")
    parser.add_argument("--padded_img_dir", type=str, help="Directory to save padded images and statistics.")
    parser.add_argument("--latent_save_dir", type=str, required=True, help="Directory to save latent representations.")
    parser.add_argument("--dcae_model_path", type=str, required=True, help="Path to pretrained DC-AE model directory.")
    parser.add_argument("--iter_idx", type=int, required=True, help="Current iteration index for parallel processing.")
    parser.add_argument("--max_iter_idx", type=int, default=4, help="Total number of parallel iterations.")
    parser.add_argument("--axis", type=int, required=True, help="Axis along which to slice (0, 1, or 2).")
    parser.add_argument("--batch_size", type=int, default=1600, help="Batch size for DC-AE encoding.")

    return parser.parse_args()

def main():
    PROBLEM_SUBS = []
    args = parse_args()

    device = torch.device("cuda")
    dc_ae: AutoencoderDC = AutoencoderDC.from_pretrained(args.dcae_model_path, torch_dtype=torch.float32).to(device).eval()

    img_dir = args.img_dir
    latent_save_dir = args.latent_save_dir
    if not os.path.exists(latent_save_dir):
        os.makedirs(latent_save_dir)
    subj_list = os.listdir(img_dir)
    subj_list = sorted(subj_list)
    iter_len = int(len(subj_list) / args.max_iter_idx)
    iter_idx = args.iter_idx
    start_idx = iter_idx * iter_len
    end_idx = (iter_idx + 1) * iter_len if iter_idx != (args.max_iter_idx-1) else len(subj_list)
    print(f"Processing from {start_idx} to {end_idx}")
    subj_count_idx = 0

    for subj in subj_list[start_idx : end_idx]:
        if os.path.exists(os.path.join(latent_save_dir, f'axis{args.axis}', subj)):
            print(f"Skip {subj} because it already exists")
            continue
        subj_count_idx += 1
        if subj_count_idx % 10 == 0:
            print(f"Processing {subj}: {subj_count_idx}/{end_idx - start_idx}")

        subj_dir = os.path.join(img_dir, subj)
        try:
            brain_list = os.listdir(subj_dir)
        except Exception as e:
            print(e, subj)
            continue
        brain_list = sorted(brain_list)
        brain_list = [brain for brain in brain_list if "frame" in brain]
        all_brain = []
        for i in range(len(brain_list)):
            brain = torch.load(os.path.join(subj_dir, f"frame_{i}.pt"), weights_only=False)
            brain = brain.squeeze()

            all_brain.append(brain)

        all_brain = torch.stack(all_brain, dim=0) # [T, D, H, W]
        ax0_len = all_brain.shape[1]
        ax1_len = all_brain.shape[2]
        ax2_len = all_brain.shape[3]

        ax0_even = (96 - ax0_len) % 2 == 0
        ax1_even = (96 - ax1_len) % 2 == 0
        ax2_even = (96 - ax2_len) % 2 == 0

        if ax0_even:
            ax0_pad_f = (96 - ax0_len) // 2
            ax0_pad_b = (96 - ax0_len) // 2

        else:
            ax0_pad_f = (96 - ax0_len) // 2 + 1
            ax0_pad_b = (96 - ax0_len) // 2

        if ax1_even:
            ax1_pad_f = (96 - ax1_len) // 2
            ax1_pad_b = (96 - ax1_len) // 2
        else:
            ax1_pad_f = (96 - ax1_len) // 2 + 1
            ax1_pad_b = (96 - ax1_len) // 2

        if ax2_even:
            ax2_pad_f = (96 - ax2_len) // 2
            ax2_pad_b = (96 - ax2_len) // 2
        else:
            ax2_pad_f = (96 - ax2_len) // 2 + 1
            ax2_pad_b = (96 - ax2_len) // 2

        try:
            assert ax1_len <= 96, f"ax1_len: {ax1_len}, subj: {subj}"
        except Exception as e:
            print(subj)
            PROBLEM_SUBS.append(subj)
            continue


        all_brain = torch.nn.functional.pad(all_brain, (ax2_pad_f, ax2_pad_b, ax1_pad_f, ax1_pad_b, ax0_pad_f, ax0_pad_b), 'constant', 0)


        assert all_brain.shape == (len(brain_list), 96, 96, 96)

        all_brain = torch.clamp(all_brain, min=0)


        if args.axis == 0:
            foreground = all_brain != 0

            global_stats = {
                'valid_voxels': foreground.sum().item(),
                'global_mean': all_brain[foreground].mean().item(),
                'global_std': all_brain[foreground].std().item(),
                'global_max': all_brain[foreground].max().item(),
                'global_min': all_brain[foreground].min().item(),
            }

            voxel_stats = {
                'voxel_mean': all_brain.mean(dim=0),
                'voxel_std': all_brain.std(dim=0),
                'voxel_max': all_brain.max(dim=0).values,
                'voxel_min': all_brain.min(dim=0).values,
            }

            os.makedirs(os.path.join(args.padded_img_dir, subj), exist_ok=True)

            torch.save(global_stats, os.path.join(args.padded_img_dir, subj, "global_stats.pt"))
            torch.save(voxel_stats, os.path.join(args.padded_img_dir, subj, "voxel_stats.pt"))

            for t in range(all_brain.shape[0]):
                if not os.path.exists(os.path.join(args.padded_img_dir, subj)):
                    os.makedirs(os.path.join(args.padded_img_dir, subj), exist_ok=True)

                assert all_brain[t].shape == (96, 96, 96), f"Unexpected tensor shape: {all_brain[t].shape} in file {os.path.join(subj_dir, f'frame_{t}.pt')}"
                np.save(os.path.join(args.padded_img_dir, subj, f"frame_{t}.npy"), all_brain[t].unsqueeze(-1).numpy())

        brain_max = torch.max(all_brain)
        all_brain = all_brain / brain_max

        all_brain = all_brain - 0.5
        all_brain = all_brain * 2

        all_brain = all_brain.float()
        all_brain = torch.stack([all_brain, all_brain, all_brain], dim=4)

        if args.axis==0:
            all_brain = all_brain.permute(0,1,4,2,3)
        elif args.axis==1:
            all_brain = all_brain.permute(0,2,4,1,3)
        elif args.axis==2:
            all_brain = all_brain.permute(0,3,4,1,2)


        all_brain = all_brain.reshape(-1,3,96,96)

        max_iter = int(all_brain.shape[0] / args.batch_size) + 1
        all_latent = torch.zeros(all_brain.shape[0],32,3,3)


        for i in range(max_iter):
            if len(all_brain[i*args.batch_size:(i+1)*args.batch_size]) == 0:
                break
            with torch.no_grad():
                latent = dc_ae.encode(all_brain[i*args.batch_size:(i+1)*args.batch_size].to(device)).latent
            all_latent[i*args.batch_size:(i+1)*args.batch_size] = latent.cpu()


        all_latent = all_latent.reshape(-1,96,32,3,3)


        latent_save_dir_brain = os.path.join(latent_save_dir, f'axis{args.axis}', subj)
        if not os.path.exists(latent_save_dir_brain):
            os.makedirs(latent_save_dir_brain)

        for i in range(len(brain_list)):
            np.save(os.path.join(latent_save_dir_brain, f"frame_{i}.npy"), all_latent[i].numpy())


    print("Process Finished!")
    print(PROBLEM_SUBS)


if __name__ == "__main__":
    main()
