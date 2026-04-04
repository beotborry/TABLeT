import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd 
from src.datasets.base_dataset import BaseDataset

class ADHDDataset(BaseDataset):
    def __init__(self, root_dir, subjects, data_type, task="ADHD", num_frames=None, frame_iid=True, num_axis=1, num_input_frames=1, slice_axis='axis0', FD=0.0,
                 random_sample_frames=False, sample_first_frame=False):
        super().__init__(
            root_dir=root_dir,
            subjects=subjects,
            data_type=data_type,
            num_frames=num_frames,
            frame_iid=frame_iid,
            num_axis=num_axis,
            num_input_frames=num_input_frames,
            slice_axis=slice_axis,

            random_sample_frames=random_sample_frames,
            sample_first_frame=sample_first_frame
        )
        self.task = task
        self.FD = FD
        
        print(len(subjects), "subjects found")
        
        if self.FD == 0:
            df_meta = pd.read_csv(os.path.join(root_dir, 'metadata', 'metadata_FD_None_MNI152_remove_prob_subs.csv'))
        elif self.FD > 0:
            raise NotImplementedError("FD > 0.0 is not implemented yet.")
        df_meta['subject_id'] = df_meta['subject_id'].astype(str)
        df_meta = df_meta[df_meta['subject_id'].isin(subjects)]
        self.df_meta = df_meta.copy()
        
        self.load_frame_paths()
        
        self.class_counts = self.df_meta['DIAGNOSIS'].value_counts()
        print("Class counts:", self.class_counts)
        self.pos_weight = self.class_counts.max() / self.class_counts.min()
        print("Positive weight for loss function:", self.pos_weight)
        
    
    def __getitem__(self, idx):
        frame_tensor, attn_mask, subject = self.get_frame_tensor(idx)        
        label = self.get_label(subject)
        TR = self.df_meta.loc[self.df_meta['subject_id'] == subject, 'TR'].values[0]
        return frame_tensor, label, (subject, TR, attn_mask)

    def get_label(self, subject):
        if self.task == "ADHD":
            label = float(self.df_meta.loc[self.df_meta['subject_id'] == subject, 'DIAGNOSIS'].values[0])
            
        else:
            label = 0

        assert label is not None, f"Label not found for subject {subject} in metadata."
        return label
    
            

def split_subjects(root_dir, seed, test_set_id, FD=0.0):
    if FD == 0.0:
        df_meta = pd.read_csv(os.path.join(root_dir, 'metadata', 'metadata_FD_None_MNI152_remove_prob_subs.csv'))
        split_dict = torch.load(os.path.join(root_dir, 'metadata', 'split_dict_FD_None_site_stratified_MNI152_remove_prob_subs.pt'))
        train_subjects = split_dict[test_set_id]['train']
        val_subjects = split_dict[test_set_id]['val']
        test_subjects = split_dict[test_set_id]['test']
    elif FD > 0.0:
        raise NotImplementedError("Splitting for FD > 0.0 is not implemented yet.")
    
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
    
    return train_subjects, val_subjects, test_subjects

def get_dataloaders(data_type, root_dir, train_ratio, val_ratio, seed, test_set_id, batch_size, num_workers, task="vae", num_frames=None, frame_iid=True, num_axis=1, num_input_frames=1, slice_axis='axis0', class_balanced=False, FD=0.0,
                    random_sample_frames=False):
    train_subjects, val_subjects, test_subjects = split_subjects(root_dir, seed, test_set_id, FD=FD)
    
    if num_frames == "None":
        num_frames = None
    train_dataset = ADHDDataset(root_dir, train_subjects, data_type, task, num_frames=num_frames, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames, slice_axis=slice_axis, FD=FD,
                                random_sample_frames=random_sample_frames)
    if class_balanced:
        sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights), replacement=True)
        shuffle = False
    else: 
        sampler = None
        shuffle = True
    
    if task == 'tff_phase_1' or task == 'tff_phase_2':
        val_dataset = ADHDDataset(root_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                slice_axis=slice_axis, FD=FD, random_sample_frames=True, sample_first_frame=True)
        test_dataset = ADHDDataset(root_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                slice_axis=slice_axis, FD=FD, random_sample_frames=True, sample_first_frame=True)
    else:
        if frame_iid:
            # num_frames == num_input_frames
            val_dataset = ADHDDataset(root_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                    slice_axis=slice_axis, FD=FD, random_sample_frames=False)
            test_dataset = ADHDDataset(root_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                        slice_axis=slice_axis, FD=FD, random_sample_frames=False)
        else:
            # num_frames can be larger than the num_input_frames
            val_dataset = ADHDDataset(root_dir, val_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                    slice_axis=slice_axis, FD=FD, random_sample_frames=False)
            test_dataset = ADHDDataset(root_dir, test_subjects, data_type, task, num_frames=None, frame_iid=frame_iid, num_axis=num_axis, num_input_frames=num_input_frames,
                                        slice_axis=slice_axis, FD=FD, random_sample_frames=False)

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
