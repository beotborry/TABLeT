import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd 
from src.datasets.base_dataset import BaseDataset

class HBNMovieClfDataset(BaseDataset):
    def __init__(self, root_dir, subjects, data_type, task="MovieClf", num_frames=None, frame_iid=True, num_axis=1, num_input_frames=1, slice_axis='axis0', FD=0.0,
                 random_sample_frames=False, first_consecutive_frames=False):
        super().__init__(
            root_dir=root_dir,
            subjects=subjects,
            data_type=data_type,
            num_frames=num_frames,
            frame_iid=frame_iid,
            num_axis=num_axis,
            num_input_frames=num_input_frames,
            slice_axis=slice_axis,

            random_sample_frames=random_sample_frames
        )
        self.task = task
        self.first_consecutive_frames = first_consecutive_frames
        
        print(len(subjects), "subjects found")
        
        if task == "MovieClf":
            df_meta = pd.read_csv(os.path.join(root_dir, 'metadata', 'metadata_FD_None_MNI152_Movie_clf_remove_prob_subs.csv'))

            df_meta['subject_id'] = df_meta['subject_id'].astype(str)
            df_meta = df_meta[df_meta['subject_id'].isin(subjects)]

            self.df_meta = df_meta.copy()
        
        self.load_frame_paths()
        print(len(self.frame_paths), "frames loaded")
        
    def load_frame_volume_paths(self):
        for subject in self.subjects:
            for movie in ['movieDM', 'movieTP']:
                subject_path = os.path.join(f"{self.root_dir}/{self.img_folder_name}/", subject, movie)

                curr_frame_paths = []
                if os.path.isdir(subject_path):
                    with os.scandir(subject_path) as entries:
                        frames = [entry.name for entry in entries 
                                if entry.is_file() and entry.name.startswith("frame")]
                    frames.sort(key=lambda fname: int(fname[6:fname.rfind('.')]))
                    
                    if self.first_consecutive_frames: # cache full frame if first consecutive frames
                        for n in range(self.num_input_frames):
                            curr_frame_paths.append((os.path.join(subject_path, frames[n]), subject, movie))
                        self.frame_paths.append(curr_frame_paths)
                        curr_frame_paths = []
                        continue
                    else:
                        if self.random_sample_frames: # cache full frame if random sampling
                            for frame in frames:
                                curr_frame_paths.append((os.path.join(subject_path, frame), subject, movie))
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

                                curr_frame_paths.append((os.path.join(subject_path, frames[n]), subject, movie))

                            if len(curr_frame_paths) == self.num_input_frames:
                                self.frame_paths.append(curr_frame_paths)
                                curr_frame_paths = []
                else:
                    print(f"Subject path {subject_path} is not a file.")
                                
                if len(curr_frame_paths) > 0 and len(curr_frame_paths) == self.num_input_frames * self.num_axis:        
                    self.frame_paths.append(curr_frame_paths)
      
      
    def load_frame_token_paths(self):
        for subject in self.subjects:
            for movie in ['movieDM', 'movieTP']:
                if self.data_type == "token": 
                    subject_path = os.path.join(f"{self.root_dir}/latents_MNI152/{self.slice_axis}_pt_new_zip", subject, movie)
                    curr_frame_paths = []
                    if os.path.isdir(subject_path):
                        if movie == 'movieDM':
                            frames = [f"frame_{n}.pt" for n in range(750)]
                        elif movie == 'movieTP':
                            frames = [f"frame_{n}.pt" for n in range(250)]
                        


                        if self.first_consecutive_frames:
                            for n in range(self.num_input_frames):
                                curr_frame_paths.append(((os.path.join(subject_path, frames[n]), subject, movie), \
                                                            (os.path.join(subject_path, frames[n]).replace("axis0", "axis1"), subject, movie), \
                                                            (os.path.join(subject_path, frames[n]).replace("axis0", "axis2"), subject, movie)))

                            self.frame_paths.append(curr_frame_paths)
                            curr_frame_paths = []
                            continue
                        
                        if self.random_sample_frames: # cache full frame if random sampling
                            for frame in frames:
                                if self.num_axis == 1:
                                    curr_frame_paths.append((os.path.join(subject_path, frame), subject, movie))
                                elif self.num_axis == 3 and self.num_input_frames > 1:
                                    curr_frame_paths.append(((os.path.join(subject_path, frame), subject, movie), \
                                                                (os.path.join(subject_path, frame).replace("axis0", "axis1"), subject, movie), \
                                                                (os.path.join(subject_path, frame).replace("axis0", "axis2"), subject, movie)))
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
                                
                                if self.num_axis == 3 and self.num_input_frames > 1:
                                    assert self.slice_axis == 'axis0', f"slice_axis {self.slice_axis} not supported for num_axis {self.num_axis}"
                                    curr_frame_paths.append(((os.path.join(subject_path, frames[n]), subject, movie), \
                                                                (os.path.join(subject_path, frames[n]).replace("axis0", "axis1"), subject, movie), \
                                                                (os.path.join(subject_path, frames[n]).replace("axis0", "axis2"), subject, movie)))
                                else:
                                    raise ValueError(f"num_axis {self.num_axis} not supported")
                            
                            if len(curr_frame_paths) == self.num_input_frames:
                                self.frame_paths.append(curr_frame_paths)
                                curr_frame_paths = []
                            elif len(curr_frame_paths) < self.num_input_frames and is_first_sample:
                                # padding
                                for n in range(len(curr_frame_paths), self.num_input_frames):
                                    if self.num_axis==1:
                                        curr_frame_paths.append(('zero', subject, movie))
                                    elif self.num_axis == 3:
                                        curr_frame_paths.append((('zero', subject, movie), ('zero', subject, movie), ('zero', subject, movie)))
                                self.frame_paths.append(curr_frame_paths)
                                curr_frame_paths = []
                            is_first_sample = False
                    else:
                        print(f"Subject path {subject_path} is not a file.")
                                    
                    if len(curr_frame_paths) > 0 and len(curr_frame_paths) == self.num_input_frames:
                        self.frame_paths.append(curr_frame_paths)


        print(f"Found {len(self.frame_paths)} frames.")
        
        
    def get_frame_tensor(self, idx):
        if self.data_type == "volume":
            if self.first_consecutive_frames:
                frame_paths = self.frame_paths[idx]
                subject = frame_paths[0][1]
                movie = frame_paths[0][2]
                global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, movie, "global_stats.pt")
                global_stats = torch.load(global_stat_path)
                
                assert len(frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(frame_paths)}"
                frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _, _ in frame_paths], dim=-1)
                attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _, _ in frame_paths], dtype=torch.bool)
                frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
                return frame_tensor, attn_mask, subject, movie
            else:
                if self.random_sample_frames:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][1]
                    movie = frame_paths[0][2]
                    global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, movie, "global_stats.pt")
                    global_stats = torch.load(global_stat_path)
                    
                    curr_max_frames = len(frame_paths)
                    random_start_idx = np.random.choice(np.arange(0, curr_max_frames - self.num_input_frames + 1, self.num_input_frames), size=1, replace=False)[0]
                    selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
                    
                    assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                    frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _, _ in selected_frame_paths], dim=-1)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _, _ in selected_frame_paths], dtype=torch.bool)
                    frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                    assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
                else:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][1]
                    movie = frame_paths[0][2]
                    global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject, movie, "global_stats.pt")
                    global_stats = torch.load(global_stat_path)
                    frame_tensor = torch.cat([torch.from_numpy(np.load(frame_path)).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _, _ in frame_paths], dim=-1)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _, _ in frame_paths], dtype=torch.bool)
                    frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                    assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
            
                return frame_tensor, attn_mask, subject, movie
        
        elif self.data_type == "token":
            if self.num_axis == 3 and self.num_input_frames > 1:
                if self.first_consecutive_frames:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][0][1]
                    movie = frame_paths[0][0][2]
                    
                    selected_frame_paths = frame_paths
                    assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                    full_frame_tensor = torch.stack([
                        torch.load(os.path.join(f"{self.root_dir}/latents_MNI152/axis{axis}_pt_new_zip", subject, movie, "full_tensor.pt")) for axis in range(3)
                    ], dim=1)[:self.num_input_frames]
                    assert full_frame_tensor.shape[0] == self.num_input_frames, f"Unexpected number of frames: {full_frame_tensor.shape[0]}"
                    frame_tensor = full_frame_tensor
                    attn_mask = torch.tensor([[frame_paths[0][0] != 'zero'] for frame_paths in selected_frame_paths], dtype=torch.bool)
                    attn_mask = attn_mask.flatten()
                    return frame_tensor, attn_mask, subject, movie
                
                if self.random_sample_frames:
                    frame_paths = self.frame_paths[idx] # [((frame0_axis0, subject, movie), (frame0_axis1, subject, movie), (frame0_axis2, subject, movie)), ((frame1_axis0, subject, movie), (frame1_axis1, subject, movie), (frame1_axis2, subject, movie)) ...]
                    subject = frame_paths[0][0][1]
                    movie = frame_paths[0][0][2]
                    curr_max_frames = len(frame_paths)
                    assert curr_max_frames == 750 or curr_max_frames == 250, f"Unexpected number of frames: {curr_max_frames}"
                    
                    if curr_max_frames <= self.num_input_frames:
                        selected_frame_paths = frame_paths
                        for _ in range(self.num_input_frames - curr_max_frames):
                            selected_frame_paths.append((('zero', subject, movie), ('zero', subject, movie), ('zero', subject, movie)))
                    else:
                        # random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                        random_start_idx = np.random.choice(np.arange(0, curr_max_frames - self.num_input_frames + 1, self.num_input_frames), size=1, replace=False)[0]
                        selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]

                    assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"

                    # frame_tensor = torch.stack([torch.stack([torch.from_numpy(np.load(frame_path, allow_pickle=False)) if frame_path != 'zero' else torch.zeros((96, 32, 3, 3)) for frame_path, _, _ in frame_paths], dim=0) for frame_paths in selected_frame_paths], dim=0)
                    full_frame_tensor = torch.stack([
                        torch.load(os.path.join(f"{self.root_dir}/latents_MNI152/axis{axis}_pt_new_zip", subject, movie, "full_tensor.pt")) for axis in range(3)
                    ], dim=1)
                    if curr_max_frames > self.num_input_frames:
                        frame_tensor = full_frame_tensor[random_start_idx:random_start_idx + self.num_input_frames, :, :, :, :, :]
                    else:
                        frame_tensor = full_frame_tensor
                        for _ in range(self.num_input_frames - len(full_frame_tensor)):
                            frame_tensor = torch.cat([frame_tensor, torch.zeros((1, 3, 96, 32, 3, 3))], dim=0)
                            
                    attn_mask = torch.tensor([[frame_paths[0][0] != 'zero'] for frame_paths in selected_frame_paths], dtype=torch.bool)
                    attn_mask = attn_mask.flatten()
                else:
                    # frame_paths = self.frame_paths[idx] # [((frame0_axis0, subject), (frame0_axis1, subject), (frame0_axis2, subject)), ((frame1_axis0, subject), (frame1_axis1, subject), (frame1_axis2, subject)) ...]
                    subject = self.frame_paths[idx][0][0][1]
                    movie = self.frame_paths[idx][0][0][2]
                    
                    full_frame_tensor = torch.stack([
                        torch.load(os.path.join(f"{self.root_dir}/latents_MNI152/axis{axis}_pt_new_zip", subject, movie, "full_tensor.pt")) for axis in range(3)
                    ], dim=1) # [full_T, 3, 96, 32, 3, 3]
                    
                    start_idx = int(self.frame_paths[idx][0][0][0][self.frame_paths[idx][0][0][0].rfind("/")+7: self.frame_paths[idx][0][0][0].rfind(".")])
                    if len(full_frame_tensor) < self.num_input_frames:
                        for _ in range(self.num_input_frames - len(full_frame_tensor)):
                            full_frame_tensor = torch.cat([full_frame_tensor, torch.zeros((1, 3, 96, 32, 3, 3))], dim=0)
                        frame_tensor = full_frame_tensor
                    else: frame_tensor = full_frame_tensor[start_idx:start_idx + self.num_input_frames]
                    
                    # frame_tensor = torch.stack([torch.stack([torch.from_numpy(np.load(frame_path, allow_pickle=False)) if frame_path != 'zero' else torch.zeros((96, 32, 3, 3)) for frame_path, _, _ in frame_paths], dim=0) for frame_paths in self.frame_paths[idx]], dim=0)
                    attn_mask = torch.tensor([[frame_paths[0][0] != 'zero'] for frame_paths in self.frame_paths[idx]], dtype=torch.bool)
                    attn_mask = attn_mask.flatten()
                    
            else: raise NotImplementedError(f"num_axis {self.num_axis} not supported")
                    
            return frame_tensor, attn_mask, subject, movie
    
                          
    def __getitem__(self, idx):
        frame_tensor, attn_mask, subject, movie = self.get_frame_tensor(idx)        
        label = int(movie == 'movieDM')
        TR = self.df_meta.loc[self.df_meta['subject_id'] == subject, 'TR'].values[0]
        return frame_tensor, label, (f'{subject}_{movie}', TR, attn_mask)
    

def split_subjects(root_dir, seed, test_set_id, FD=0.0, task='ADHD'):
    if task == "MovieClf":
        split_dict = torch.load(os.path.join(root_dir, 'metadata', 'split_FD_None_MNI152_Movie_clf_remove_prob_subs_site_diag.pt'))
    else: raise NotImplementedError(f"Task {task} not implemented yet.")
    train_subjects = split_dict[test_set_id]['train']
    val_subjects = split_dict[test_set_id]['val']
    test_subjects = split_dict[test_set_id]['test']

    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
    
    return train_subjects, val_subjects, test_subjects

def get_dataloaders(data_type, root_dir, train_ratio, val_ratio, seed, test_set_id, batch_size, num_workers, task="vae", num_frames=None, frame_iid=True, num_axis=1, num_input_frames=1, slice_axis='axis0', class_balanced=False, FD=0.0,
                    random_sample_frames=False, first_consecutive_frames=False):
    train_subjects, val_subjects, test_subjects = split_subjects(root_dir, seed, test_set_id, FD=FD, task=task)
    
    if num_frames == "None":
        num_frames = None
    train_dataset = HBNMovieClfDataset(root_dir, train_subjects, data_type, task, num_frames=num_frames, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames, slice_axis=slice_axis, FD=FD,
                                random_sample_frames=random_sample_frames, first_consecutive_frames=first_consecutive_frames)
    if class_balanced:
        sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights), replacement=True)
        shuffle = False
    else: 
        sampler = None
        shuffle = True
    
    if frame_iid:
        # num_frames == num_input_frames
        val_dataset = HBNMovieClfDataset(root_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                   slice_axis=slice_axis, FD=FD, random_sample_frames=False, first_consecutive_frames=first_consecutive_frames)
        test_dataset = HBNMovieClfDataset(root_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                    slice_axis=slice_axis, FD=FD, random_sample_frames=False, first_consecutive_frames=first_consecutive_frames)
    else:
        # num_frames can be larger than the num_input_frames
        val_dataset = HBNMovieClfDataset(root_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                   slice_axis=slice_axis, FD=FD, random_sample_frames=False, first_consecutive_frames=first_consecutive_frames)
        test_dataset = HBNMovieClfDataset(root_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                    slice_axis=slice_axis, FD=FD, random_sample_frames=False, first_consecutive_frames=first_consecutive_frames)

    persistent_workers = num_workers > 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent_workers, drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=persistent_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=persistent_workers, drop_last=False)

    return train_loader, val_loader, test_loader
