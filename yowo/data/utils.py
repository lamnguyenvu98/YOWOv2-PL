import torch

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