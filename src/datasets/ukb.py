import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd 
from src.datasets.base_dataset import BaseDataset
from nilearn import connectome

class UKBDataset(BaseDataset):
    def __init__(self, root_dir, img_dir, subjects, data_type, task="gender", num_frames=None, frame_iid=True, num_axis=1, num_input_frames=1, slice_axis='axis0',
                 random_sample_frames=False, test_set_id=0):
        super().__init__(
            root_dir=root_dir,
            subjects=subjects,
            data_type=data_type,
            num_frames=num_frames,
            frame_iid=frame_iid,
            num_axis=num_axis,
            num_input_frames=num_input_frames,
            slice_axis=slice_axis,

            img_folder_name='img',
            random_sample_frames=random_sample_frames
        )
        self.task = task
        self.img_dir = img_dir
        self.test_set_id = test_set_id

        print(len(subjects), "subjects found")

        df_meta = pd.read_csv(os.path.join(root_dir, 'metadata', 'ukb_1_metadata.csv'))
        df_meta['subject_id'] = df_meta['subject_id'].astype(str)
        df_meta = df_meta[df_meta['subject_id'].isin(subjects)]
        print(f"Number of subjects in metadata: {len(df_meta)}")
        if len(df_meta) == 0:
            raise ValueError("No subjects found in metadata. Please check the subjects list and metadata file.")
        self.df_meta = df_meta.copy()
        
        if task == "age" or task == "intelligence":
            self.df_meta[task] = self.df_meta[task].astype(float)
            self.global_mean = self.df_meta[task].mean()
            self.global_std = self.df_meta[task].std()


        self.load_frame_paths()

    def load_frame_volume_paths(self):
        for subject in self.subjects:
            subject_name = self.df_meta.loc[self.df_meta["subject_id"] == subject, "subject_dir"].values[0]
            subject_path = os.path.join(f"{self.img_dir}/{self.img_folder_name}/", subject_name)

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
            subject_dir = self.df_meta.loc[self.df_meta["subject_id"] == subject, "subject_dir"].values[0]
            subject_path = os.path.join(f"{self.root_dir}/latents_MNI152/{self.slice_axis}", subject_dir)

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
            subject_dir = self.df_meta.loc[self.df_meta["subject_id"] == subject, "subject_dir"].values[0]
            roi_signals_path = os.path.join(f"{self.root_dir}/roi_signals_MNI152/", subject_dir, f"{subject_dir}_roi_signals.csv")
            if not os.path.exists(roi_signals_path):
                raise FileNotFoundError(f"ROI signals file {roi_signals_path} not found.")
            self.frame_paths.append([roi_signals_path, subject])

    def get_frame_tensor(self, idx):
        if self.data_type == "volume":
            if self.task == 'swift_pretraining':
                frame_paths = self.frame_paths[idx]
                subject = frame_paths[0][1]
                subject_name = self.df_meta.loc[self.df_meta["subject_id"] == subject, "subject_dir"].values[0]
                global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject_name, "global_stats.pt")
                global_stats = torch.load(global_stat_path)

                curr_max_frames = len(frame_paths)
                if curr_max_frames <= self.num_input_frames:
                    selected_frame_paths = frame_paths
                    for _ in range(self.num_input_frames - curr_max_frames):
                        selected_frame_paths.append(('zero', subject))
                else:
                    random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                    selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
    
                assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                full_range = np.arange(0, curr_max_frames - self.num_input_frames)
                excluded_range = np.arange(random_start_idx - self.num_input_frames, random_start_idx + self.num_input_frames)
                available_range = np.setdiff1d(full_range, excluded_range)
                random_start_idx_2 = np.random.choice(available_range, size=1, replace=False)[0]
                selected_frame_paths_2 = frame_paths[random_start_idx_2:random_start_idx_2 + self.num_input_frames]
                assert len(selected_frame_paths_2) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths_2)}"
                
                
                frame_tensor = torch.cat([torch.load(frame_path).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _ in selected_frame_paths], dim=-1)
                attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in selected_frame_paths], dtype=torch.bool)
                frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                background_value = torch.min(frame_tensor)
                frame_tensor = torch.nn.functional.pad(frame_tensor, (0, 0, 6, 6, 0, 0, 9, 9), value=background_value)
                assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
                
                frame_tensor_2 = torch.cat([torch.load(frame_path).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _ in selected_frame_paths_2], dim=-1)
                frame_tensor_2 = (frame_tensor_2 - global_stats['global_mean']) / global_stats['global_std']
                background_value = torch.min(frame_tensor_2)
                frame_tensor_2 = torch.nn.functional.pad(frame_tensor_2, (0, 0, 6, 6, 0, 0, 9, 9), value=background_value)
                assert frame_tensor_2.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor_2.shape} in file {frame_paths}"
                
                return (frame_tensor, frame_tensor_2), attn_mask, subject
            else:
                if self.random_sample_frames:
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][1]
                    subject_name = self.df_meta.loc[self.df_meta["subject_id"] == subject, "subject_dir"].values[0]
                    global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject_name, "global_stats.pt")
                    global_stats = torch.load(global_stat_path)
                    
                    curr_max_frames = len(frame_paths)
                    if curr_max_frames <= self.num_input_frames:
                        selected_frame_paths = frame_paths
                        for _ in range(self.num_input_frames - curr_max_frames):
                            selected_frame_paths.append(('zero', subject))
                    else:
                        random_start_idx = np.random.randint(0, curr_max_frames - self.num_input_frames + 1)
                        selected_frame_paths = frame_paths[random_start_idx:random_start_idx + self.num_input_frames]
                    
                    assert len(selected_frame_paths) == self.num_input_frames, f"Unexpected number of frames: {len(selected_frame_paths)}"
                    frame_tensor = torch.cat([torch.load(frame_path).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _ in selected_frame_paths], dim=-1)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in selected_frame_paths], dtype=torch.bool)
                    frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                    background_value = torch.min(frame_tensor)
                    frame_tensor = torch.nn.functional.pad(frame_tensor, (0, 0, 6, 6, 0, 0, 9, 9), value=background_value)
                    assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
                else:        
                    frame_paths = self.frame_paths[idx]
                    subject = frame_paths[0][1]
                    subject_name = self.df_meta.loc[self.df_meta["subject_id"] == subject, "subject_dir"].values[0]
                    global_stat_path = os.path.join(self.root_dir, self.img_folder_name, subject_name, "global_stats.pt")
                    global_stats = torch.load(global_stat_path)
                    frame_tensor = torch.cat([torch.load(frame_path).permute(3, 0, 1, 2).unsqueeze(-1) for frame_path, _ in frame_paths], dim=-1)
                    attn_mask = torch.tensor([True if frame_path != 'zero' else False for frame_path, _ in frame_paths], dtype=torch.bool)
                    frame_tensor = (frame_tensor - global_stats['global_mean']) / global_stats['global_std']
                    background_value = torch.min(frame_tensor)
                    frame_tensor = torch.nn.functional.pad(frame_tensor, (0, 0, 6, 6, 0, 0, 9, 9), value=background_value)
                    assert frame_tensor.shape == (1, 96, 96, 96, self.num_input_frames), f"Unexpected tensor shape: {frame_tensor.shape} in file {frame_paths}"
                    
        elif self.data_type == 'roi_signals':
            roi_signals_path, subject = self.frame_paths[idx]
            roi_signals_df = pd.read_csv(roi_signals_path)
            signals = roi_signals_df.iloc[:, 1:].values  # (T, N)
            
            stat_dict = torch.load(os.path.join(self.root_dir, 'metadata', f'UKB_tid{self.test_set_id}_stat_dict.pt'))
            
            signals = signals.T[:, :490]
            signals = (signals - stat_dict['medians'][:, None]) / (stat_dict['iqrs'][:, None] + 1e-6)

            
            # from BrainJEPA impl.
            seq_length = 490 
            clip_size = 3 * 160
            start_idx, end_idx = self._get_start_end_idx(seq_length, clip_size)
            signals = torch.from_numpy(signals)
            ts_array = self._temporal_sampling(
                signals, start_idx, end_idx, 160
            )
            frame_tensor = torch.unsqueeze(ts_array, 0).to(torch.float32)
            attn_mask = torch.tensor(0)
             
        return frame_tensor, attn_mask, subject

    def __getitem__(self, idx):
        if self.data_type == "volume" or self.data_type == "roi_signals":
            frame_tensor, attn_mask, subject = self.get_frame_tensor(idx)
        else:
            frame_tensor, attn_mask, subject = super().get_frame_tensor(idx)
        label = self.get_label(subject)
        if self.task == "age" or self.task == "intelligence":
            label = (label - self.global_mean) / (self.global_std + 1e-6)
        TR = self.df_meta.loc[self.df_meta['subject_id'] == subject, 'TR'].values[0]
        return frame_tensor, label, (subject, TR, attn_mask)

    def get_label(self, subject):
        label = float(self.df_meta.loc[self.df_meta['subject_id'] == subject, self.task].values[0])

        assert label is not None, f"Label not found for subject {subject} in metadata."
        return label
    
    def _get_start_end_idx(self, fmri_size, clip_size):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx
    
    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames

def split_subjects(root_dir, seed, test_set_id, task):
    if task == "gender" or task == "age":
        split_dict = torch.load(os.path.join(root_dir, 'metadata', 'split_by_gender_age.pt'))
        train_subjects = split_dict[test_set_id]['train']
        val_subjects = split_dict[test_set_id]['val']
        test_subjects = split_dict[test_set_id]['test']
        
        print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
        
        return train_subjects, val_subjects, test_subjects

    elif task == "intelligence":
        split_dict = torch.load(os.path.join(root_dir, 'metadata', 'split_by_gender_age_intelligence.pt'))
        train_subjects = split_dict[test_set_id]['train']
        val_subjects = split_dict[test_set_id]['val']
        test_subjects = split_dict[test_set_id]['test']
        
        print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
        
        return train_subjects, val_subjects, test_subjects
    
    else:
        raise ValueError(f"Task {task} not supported.")
    
def get_dataloaders(data_type, root_dir, img_dir, train_ratio, val_ratio, seed, test_set_id, batch_size, num_workers, task="vae", num_frames=None, frame_iid=True, num_axis=1, num_input_frames=1, slice_axis='axis0', class_balanced=False, FD=0.0,
                    random_sample_frames=False):
    train_subjects, val_subjects, test_subjects = split_subjects(root_dir, seed, test_set_id, task)
    
    if num_frames == "None":
        num_frames = None
    train_dataset = UKBDataset(root_dir, img_dir, train_subjects, data_type, task, num_frames=num_frames, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames, slice_axis=slice_axis,
                                random_sample_frames=random_sample_frames, test_set_id=test_set_id)

    if class_balanced:
        sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights), replacement=True)
        shuffle = False
    else: 
        sampler = None
        shuffle = True
    
    if frame_iid:
        # num_frames == num_input_frames
        val_dataset = UKBDataset(root_dir, img_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                   slice_axis=slice_axis, random_sample_frames=False, test_set_id=test_set_id)
        test_dataset = UKBDataset(root_dir, img_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                    slice_axis=slice_axis, random_sample_frames=False, test_set_id=test_set_id)
    else:
        # num_frames can be larger than the num_input_frames
        val_dataset = UKBDataset(root_dir, img_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                   slice_axis=slice_axis, random_sample_frames=False, test_set_id=test_set_id)
        test_dataset = UKBDataset(root_dir, img_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                    slice_axis=slice_axis, random_sample_frames=False, test_set_id=test_set_id)
    
    

    if task == "age" or task == "intelligence":
        train_mean = train_dataset.global_mean
        train_std = train_dataset.global_std
        print(f"Train mean: {train_mean}, Train std: {train_std}")

        val_dataset.global_mean = train_mean
        val_dataset.global_std = train_std
        test_dataset.global_mean = train_mean
        test_dataset.global_std = train_std

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