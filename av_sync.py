from __future__ import division

import numpy as np
from scipy.interpolate import interp2d


def inc_fps(frames, target_len):
    x = np.arange(frames.shape[1])
    y = np.arange(frames.shape[0])
    y_inc = np.linspace(0, len(frames) * (1 - 1 / target_len), target_len)
    
    return interp2d(x, y, frames, kind='linear')(x, y_inc)


def sync_audio_visual_features(mask_filename, video_filename, tot_frames=None, min_frames=None, pad='start'):
    """
    Upsample video frames to the same frame rate of mask.
    tot_frames is the number of frames of full video.
    min_frames is the minimum number of video frames required (otherwise the video is overly corrupted).
    pad is used to add padding at start or at end when the number of vifro frames is smaller than tot_frames
    """
    # Get the binary mask
    mask = np.load(mask_filename)
    
    # Get face landmarks
    video_features = np.loadtxt(video_filename, dtype=np.int32)

    # Skip highly corrupted files and correct other ones
    if len(video_features.shape) != 2 or (min_frames is not None and video_features.shape[0] < min_frames):
        return (None, None)
    elif tot_frames is not None and video_features.shape[0] < tot_frames:
        # Replication of first frame at beginning
        n_rep = tot_frames - video_features.shape[0]
        if pad == 'start':
            video_features = np.vstack((np.tile(video_features[0], (n_rep, 1)), video_features))
        elif pad == 'end':
            video_features = np.vstack((video_features, np.tile(video_features[0], (n_rep, 1))))

    video_features = inc_fps(video_features, len(mask))

    # Check dimensions
    if len(mask) == len(video_features):
        return mask.astype(np.float32), video_features
    else:
        return (None, None)

