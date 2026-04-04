import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from nilearn import connectome


class BaseDataset(Dataset):
    def __init__(self, root_dir, subjects, data_type,
                 num_frames=None, frame_iid=False,
                 num_axis=1, num_input_frames=1, 
                 slice_axis='axis0',
                 img_folder_name='padded_img_MNI152',
                 random_sample_frames=False,
                 sample_first_frame=False):
        
        self.frame_paths = []
        self.root_dir = root_dir
        self.subjects = subjects
        self.data_type = data_type
        self.num_frames = num_frames
        self.frame_iid = frame_iid
        self.num_axis = num_axis
        self.num_input_frames = num_input_frames
        self.slice_axis = slice_axis
        self.stride = self.num_input_frames
        self.img_folder_name = img_folder_name
        self.random_sample_frames = random_sample_frames
        self.sample_first_frame = sample_first_frame

        DATA_TYPE_TO_LOAD_FUNC = {
            "token": self.load_frame_token_paths,
            "volume": self.load_frame_volume_paths,
            "volume_tff_pretrain": self.load_frame_volume_paths,
            "functional_connectivity": self.load_frame_paths_as_roi_signals,
            "roi_signals": self.load_frame_paths_as_roi_signals,
        }
        self.load_frame_paths = DATA_TYPE_TO_LOAD_FUNC.get(data_type, None)

    
    def load_frame_volume_paths(self):
        for subject in self.subjects:
            subject_path = os.path.join(f"{self.root_dir}/{self.img_folder_name}/", subject)

            curr_frame_paths = []
            if os.path.isdir(subject_path):
                with os.scandir(subject_path) as entries:
                    frames = [entry.name for entry in entries 
                            if entry.is_file() and entry.name.startswith("frame")]
                frames.sort(key=lambda fname: int(fname[6:fname.rfind('.')]))
                
                if self.random_sample_frames: # cache full frame if random sampling
                    for frame in frames:
                        curr_frame_paths.append((os.path.join(subject_path, frame), subject))
                    self.frame_paths.append(curr_frame_paths)
                    curr_frame_paths = []
                    continue
                
                if self.num_frames is None: num_frames_limit = len(frames)
                else: num_frames_limit = self.num_frames             

                start_frame = 0
                for start_frame in range(start_frame, len(frames), self.stride):
                    for n in range(start_frame, start_frame + self.num_input_frames):
                        if n >= len(frames):
                            break
                        elif n >= num_frames_limit:
                            break

                        curr_frame_paths.append((os.path.join(subject_path, frames[n]), subject))

                    if len(curr_frame_paths) == self.num_input_frames:
                        self.frame_paths.append(curr_frame_paths)
                        curr_frame_paths = []
            else:
                print(f"Subject path {subject_path} is not a file.")
                            
            if len(curr_frame_paths) > 0 and len(curr_frame_paths) == self.num_input_frames * self.num_axis:        
                self.frame_paths.append(curr_frame_paths)   
                      
    def load_frame_token_paths(self):
        for subject in self.subjects:
            subject_path = os.path.join(f"{self.root_dir}/latents_MNI152/{self.slice_axis}", subject)
            is_first_sample = True
            
            if self.frame_iid:
                raise NotImplementedError("frame_iid not implemented for token data type")
            else:
                curr_frame_paths = []
                if os.path.isdir(subject_path):
                    with os.scandir(subject_path) as entries:
                        frames = [entry.name for entry in entries 
                                if entry.is_file() and entry.name.startswith("frame")]
                    frames.sort(key=lambda fname: int(fname[6:fname.rfind('.')]))
                    
                    if self.random_sample_frames: # cache full frame if random sampling
                        for frame in frames:
                            if self.num_axis == 1:
                                curr_frame_paths.append((os.path.join(subject_path, frame), subject))
                            elif self.num_axis == 3 and self.num_input_frames > 1:
                                curr_frame_paths.append(((os.path.join(subject_path, frame), subject), \
                                                            (os.path.join(subject_path, frame).replace("axis0", "axis1"), subject), \
                                                            (os.path.join(subject_path, frame).replace("axis0", "axis2"), subject)))
                            else:
                                raise ValueError(f"num_axis {self.num_axis} not supported")
                        self.frame_paths.append(curr_frame_paths)
                        curr_frame_paths = []
                        continue
                    
                    if self.num_frames is None: num_frames_limit = len(frames)
                    else: num_frames_limit = self.num_frames             
                           
                    start_frame = 0

                    for start_frame in range(start_frame, len(frames), self.stride):
                        for n in range(start_frame, start_frame + self.num_input_frames):
                            if n >= len(frames):
                                break
                            elif n >= num_frames_limit:
                                break

                            if self.num_axis == 1:
                                curr_frame_paths.append((os.path.join(subject_path, frames[n]), subject))
                            elif self.num_axis == 3 and self.num_input_frames > 1:
                                assert self.slice_axis == 'axis0', f"slice_axis {self.slice_axis} not supported for num_axis {self.num_axis}"
                                curr_frame_paths.append(((os.path.join(subject_path, frames[n]), subject), \
                                                            (os.path.join(subject_path, frames[n]).replace("axis0", "axis1"), subject), \
                                                            (os.path.join(subject_path, frames[n]).replace("axis0", "axis2"), subject)))
                            else:
                                raise ValueError(f"num_axis {self.num_axis} not supported")
                        
                        if len(curr_frame_paths) == self.num_input_frames:
                            self.frame_paths.append(curr_frame_paths)
                            curr_frame_paths = []
                        elif len(curr_frame_paths) < self.num_input_frames and is_first_sample:
                            # padding
                            for n in range(len(curr_frame_paths), self.num_input_frames):
                                if self.num_axis==1:
                                    curr_frame_paths.append(('zero', subject))
                                elif self.num_axis == 3:
                                    curr_frame_paths.append((('zero', subject), ('zero', subject), ('zero', subject)))
                            self.frame_paths.append(curr_frame_paths)
                            curr_frame_paths = []
                        is_first_sample = False
                else:
                    print(f"Subject path {subject_path} is not a file.")
                                
                if len(curr_frame_paths) > 0 and len(curr_frame_paths) == self.num_input_frames:
                    self.frame_paths.append(curr_frame_paths)


        print(f"Found {len(self.frame_paths)} frames.")

    def load_frame_paths_as_roi_signals(self):
        for subject in self.subjects:
            roi_signals_path = os.path.join(f"{self.root_dir}/roi_signals_MNI152/", subject, f"{subject}_roi_signals.csv")
            if not os.path.exists(roi_signals_path):
                raise FileNotFoundError(f"ROI signals file {roi_signals_path} not found.")
            self.frame_paths.append([roi_signals_path, subject])

    def get_frame_tensor(self, idx):
        if self.data_type == "token":
            if self.num_axis == 3 and self.num_input_frames > 1:
                if self.random_sample_frames:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][0][1]
                    curr_max_frames = len(frame_paths)
                    
                    if curr_max_frames <= self.num_input_frames:
                        selected_frame_paths = frame_paths[:-1]
                        for _ in range(self.num_input_frames - curr_max_frames + 1):
                            selected_frame_paths.append((('zero', subject), ('zero', subject), ('zero', subject)))
                    else:
                        random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                        selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
                        
                    assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                    frame_tensor = torch.stack([torch.stack([torch.from_numpy(np.load(frame_path, allow_pickle=False)) if frame_path != 'zero' else torch.zeros((96, 32, 3, 3)) for (frame_path, _) in frame_paths], dim=0) for frame_paths in selected_frame_paths], dim=0)
                    attn_mask = torch.tensor([[frame_paths[0][0] != 'zero'] for frame_paths in selected_frame_paths], dtype=torch.bool)
                    attn_mask = attn_mask.flatten()
                else:
                    # frame_paths = self.frame_paths[idx] # [((frame0_axis0, subject), (frame0_axis1, subject), (frame0_axis2, subject)), ((frame1_axis0, subject), (frame1_axis1, subject), (frame1_axis2, subject)) ...]
                    subject = self.frame_paths[idx][0][0][1]
                    frame_tensor = torch.stack([torch.stack([torch.from_numpy(np.load(frame_path, allow_pickle=False)) if frame_path != 'zero' else torch.zeros((96, 32, 3, 3)) for frame_path, _ in frame_paths], dim=0) for frame_paths in self.frame_paths[idx]], dim=0)
                    attn_mask = torch.tensor([[frame_paths[0][0] != 'zero'] for frame_paths in self.frame_paths[idx]], dtype=torch.bool)
                    attn_mask = attn_mask.flatten()
            else:
                if self.random_sample_frames:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][1]
                    curr_max_frames = len(frame_paths)
                    if curr_max_frames <= self.num_input_frames:
                        selected_frame_paths = frame_paths[:-1]
                        for _ in range(self.num_input_frames - curr_max_frames + 1):
                            selected_frame_paths.append(('zero', subject))
                    else:
                        random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                        selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
                    frame_tensor = torch.cat([
                        torch.from_numpy(np.load(frame_path)) if frame_path != 'zero' else torch.zeros((96, 32, 3, 3))
                        for frame_path, _ in selected_frame_paths
                    ], dim=0)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in selected_frame_paths], dtype=torch.bool)
                else:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][1]
                    frame_tensor = torch.cat([
                        torch.from_numpy(np.load(frame_path)) if frame_path != 'zero' else torch.zeros((96, 32, 3, 3))
                        for frame_path, _ in frame_paths
                    ], dim=0)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in frame_paths], dtype=torch.bool)
                    
        elif self.data_type == "volume":
            if self.random_sample_frames:
                frame_paths = self.frame_paths[idx]
                subject = frame_paths[0][1]
                global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, "global_stats.pt")
                global_stats = torch.load(global_stat_path)
                
                curr_max_frames = len(frame_paths)
                if curr_max_frames <= self.num_input_frames:
                    selected_frame_paths = frame_paths[:-1]
                    for _ in range(self.num_input_frames - curr_max_frames + 1):
                        selected_frame_paths.append(('zero', subject))
                else:
                    random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                    selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
                
                assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) if frame_path != 'zero' else torch.zeros((1, 96, 96, 96, 1)) for frame_path, _ in selected_frame_paths], dim=-1)
                
                attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in selected_frame_paths], dtype=torch.bool)
                frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
            else:
                frame_paths = self.frame_paths[idx]
                subject = frame_paths[0][1]
                global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, "global_stats.pt")
                global_stats = torch.load(global_stat_path)
                frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) if frame_path != 'zero' else torch.zeros((1, 96, 96, 96, 1)) for frame_path, _ in frame_paths], dim=-1)
                attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in frame_paths], dtype=torch.bool)
                frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
                
        elif self.data_type == "volume_tff_pretrain":
            if self.random_sample_frames:
                frame_paths = self.frame_paths[idx]
                subject = frame_paths[0][1]
                global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, "global_stats.pt")
                global_stats = torch.load(global_stat_path)
                voxel_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, "voxel_stats.pt")
                voxel_stats = torch.load(voxel_stat_path)
                
                curr_max_frames = len(frame_paths)
                
                if not self.sample_first_frame:
                    if curr_max_frames <= self.num_input_frames:
                        selected_frame_paths = frame_paths[:-1]
                        for _ in range(self.num_input_frames - curr_max_frames + 1):
                            selected_frame_paths.append(('zero', subject))
                    else:
                        random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                        selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
                
                else:
                    if self.num_input_frames == 1: selected_frame_paths = [frame_paths[0]]
                    else: selected_frame_paths = frame_paths[0:0+self.num_input_frames]

                assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                
                
                if len(selected_frame_paths) == 1:
                    frame_tensor = torch.from_numpy(np.load(selected_frame_paths[0][0])).permute(3, 0, 1, 2).unsqueeze(-1)
                    attn_mask = torch.tensor([True], dtype=torch.bool)
                else:
                    frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _ in selected_frame_paths], dim=-1)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in selected_frame_paths], dtype=torch.bool)
                    
                frame_tensor_global = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                assert frame_tensor_global.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor_global.shape} in file {frame_paths}"
                assert not frame_tensor_global.isnan().any(), "NaN in global tensor"


                frame_tensor_voxel = (frame_tensor - voxel_stats['voxel_mean'].unsqueeze(-1)) / (voxel_stats['voxel_std'].unsqueeze(-1) + 1e-6)
                assert frame_tensor_voxel.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor_voxel.shape} in file {frame_paths}"
                assert not frame_tensor_voxel.isnan().any(), "NaN in voxel tensor"

                frame_tensor = torch.cat([frame_tensor_global, frame_tensor_voxel], dim=0)
                assert frame_tensor.shape == (2, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
            else:
                frame_paths = self.frame_paths[idx]
                subject = frame_paths[0][1]
                global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, "global_stats.pt")
                global_stats = torch.load(global_stat_path)
                voxel_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, "voxel_stats.pt")
                voxel_stats = torch.load(voxel_stat_path)
                
                frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _ in frame_paths], dim=-1)
                attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in frame_paths], dtype=torch.bool)
                
                frame_tensor_global = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                assert frame_tensor_global.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor_global.shape} in file {frame_paths}"
                assert not frame_tensor_global.isnan().any(), "NaN in global tensor"

                frame_tensor_voxel = (frame_tensor - voxel_stats['voxel_mean'].unsqueeze(-1)) / (voxel_stats['voxel_std'].unsqueeze(-1) + 1e-6)
                assert frame_tensor_voxel.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor_voxel.shape} in file {frame_paths}"
                assert not frame_tensor_voxel.isnan().any(), "NaN in voxel tensor"

                frame_tensor = torch.cat([frame_tensor_global, frame_tensor_voxel], dim=0)
                assert frame_tensor.shape == (2, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
        
        elif self.data_type == "functional_connectivity":
            roi_signals_path, subject = self.frame_paths[idx]
            roi_signals_df = pd.read_csv(roi_signals_path)
            signals = roi_signals_df.iloc[:, 1:].values  # (T, N)

            connectivity_measure = connectome.ConnectivityMeasure(kind="correlation", standardize=True)
            connectivity = connectivity_measure.fit_transform([signals])
            return connectivity, torch.tensor(0), subject

        return frame_tensor, attn_mask, subject
        

    
    def __len__(self):
        return len(self.frame_paths)