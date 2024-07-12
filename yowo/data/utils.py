from typing import Any
import torch
from scipy.io import loadmat
import os
import time
import numpy as np

__all__ = [
    "collate_fn",
    "load_ground_truth_ucf24"
]

def load_ground_truth_ucf24(
    data_root: str,
    gt_file: str = 'splitfiles/finalAnnots.mat',
    testlist: str = 'splitfiles/testlist01.txt',
):
    gt_file = os.path.join(data_root, 'splitfiles/finalAnnots.mat')
    testlist = os.path.join(data_root, 'splitfiles/testlist01.txt')

    gt_data = loadmat(gt_file)['annot']
    n_videos = gt_data.shape[1]
    print('loading gt tubes ...')

    video_testlist = []
    with open(testlist, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            video_testlist.append(line)

    gt_videos = {}

    t1 = time.perf_counter()
    for i in range(n_videos):
        video_name = gt_data[0][i][1][0]
        if video_name not in video_testlist:
            continue
        n_tubes = len(gt_data[0][i][2][0])
        v_annotation = {}
        all_gt_boxes = []
        for j in range(n_tubes):  
            gt_one_tube = [] 
            tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
            tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
            tube_class = gt_data[0][i][2][0][j][2][0][0]
            tube_data = gt_data[0][i][2][0][j][3]
            tube_length = tube_end_frame - tube_start_frame + 1
        
            for k in range(tube_length):
                gt_boxes = []
                gt_boxes.append(int(tube_start_frame.astype(np.uint16)+k))
                gt_boxes.append(float(tube_data[k][0]))
                gt_boxes.append(float(tube_data[k][1]))
                gt_boxes.append(float(tube_data[k][0]) + float(tube_data[k][2]))
                gt_boxes.append(float(tube_data[k][1]) + float(tube_data[k][3]))
                gt_one_tube.append(gt_boxes)
            all_gt_boxes.append(gt_one_tube)

        v_annotation['gt_classes'] = tube_class
        v_annotation['tubes'] = all_gt_boxes
        gt_videos[video_name] = v_annotation

    print(time.perf_counter() - t1)
    print(len(gt_videos))
    return gt_videos

class CollateFn:
    def __call__(self, batch: Any) -> torch.Any:
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target

def collate_fn(batch):
    batch_frame_id = []
    batch_key_target = []
    batch_video_clips = []

    for sample in batch:
        key_frame_id = sample[0]
        video_clip = sample[1]
        key_target = sample[2]
        
        batch_frame_id.append(key_frame_id)
        batch_video_clips.append(video_clip)
        batch_key_target.append(key_target)

    # List [B, 3, T, H, W] -> [B, 3, T, H, W]
    batch_video_clips = torch.stack(batch_video_clips)
    
    return batch_frame_id, batch_video_clips, batch_key_target

# class CollateFunc(object):
#     def __call__(self, batch):
#         batch_frame_id = []
#         batch_key_target = []
#         batch_video_clips = []

#         for sample in batch:
#             key_frame_id = sample[0]
#             video_clip = sample[1]
#             key_target = sample[2]
            
#             batch_frame_id.append(key_frame_id)
#             batch_video_clips.append(video_clip)
#             batch_key_target.append(key_target)

#         # List [B, 3, T, H, W] -> [B, 3, T, H, W]
#         batch_video_clips = torch.stack(batch_video_clips)
        
#         return batch_frame_id, batch_video_clips, batch_key_target