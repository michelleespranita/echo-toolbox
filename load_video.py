import numpy as np
import os
from decord import VideoReader, cpu

def loadvideo_decord(
    fname,
    mode='val',
    frame_sample_rate=4,
    clip_len=16,
    must_include_frame_ids=[],
    repeat_or_pad='repeat'
):

    # load videoreader
    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []
    
    repeat_or_pad_video_ids = [] # to store the repeated/padded indices in the final video, for attention masks
    # select frame indices to load from video
    if frame_sample_rate == 'uniform':
        if clip_len > len(vr): 
            frame_id_list = [x for x in range(0, len(vr))]
            if repeat_or_pad == 'repeat': # Repeat the last frame if the video is shorter than clip_len
                i = len(frame_id_list)
                while len(frame_id_list) < clip_len:
                    frame_id_list.append(frame_id_list[-1])
                    repeat_or_pad_video_ids.append(i)
                    i += 1
        else: # Uniform sampling
            frame_id_list = np.linspace(0, len(vr) - 1, num=clip_len, dtype=int)

            for fid in must_include_frame_ids:
                if fid in frame_id_list:
                    continue
                closest_idx = np.argmin(np.abs(frame_id_list - fid))
                frame_id_list[closest_idx] = fid

            frame_id_list = frame_id_list.tolist()

    else:
        if not mode == 'train': # buffer has to have at least clip_len frames, but is allowed to have more. sampling_rate is applied.
            frame_id_list = [x for x in range(0, len(vr), frame_sample_rate)]
            if repeat_or_pad == 'repeat':
                i = len(frame_id_list)
                while len(frame_id_list) < clip_len:
                    frame_id_list.append(frame_id_list[-1])
                    repeat_or_pad_video_ids.append(i)
                    i += 1
            frame_id_list = frame_id_list[:clip_len] # this forces buffer to have clip_len frames
        
        else: # buffer has to have exactly clip_len frames. sampling_rate is applied.
            converted_len = int(clip_len * frame_sample_rate)
            seg_len = len(vr)

            frame_id_list = []
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // frame_sample_rate)
                if repeat_or_pad == 'repeat':
                    repeat_or_pad_video_ids = [len(index) + i for i in range(clip_len - seg_len // frame_sample_rate)]
                    index = np.concatenate((index, np.ones(clip_len - seg_len // frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            
            index = list(map(int, index))
            frame_id_list.extend(index)
        
    # load video
    vr.seek(0)
    buffer = vr.get_batch(frame_id_list).asnumpy()

    # pad video if necessary
    if repeat_or_pad == 'pad':
        T, C, H, W = buffer.shape
        if T < clip_len:
            missing_num_frames = clip_len - T
            padding = np.zeros((missing_num_frames, C, H, W)).astype(buffer.dtype)
            buffer = np.concatenate((buffer, padding), axis=0)
            repeat_or_pad_video_ids = [len(frame_id_list) + i for i in range(missing_num_frames)]
    
    return {
        'video': buffer,
        'frame_id': frame_id_list,
        'repeat_or_pad_video_id': repeat_or_pad_video_ids,
        'num_repeat_or_pad_frames': len(repeat_or_pad_video_ids)
    }
    
def loadvideo_decord_full(sample):
    """
    Output: numpy array (T, H, W, C)
    """
    fname = sample

    if not (os.path.exists(fname)):
        return []

    # avoid hanging issue
    if os.path.getsize(fname) < 1 * 1024:
        print('SKIP: ', fname, " - ", os.path.getsize(fname))
        return []
    try:
        vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    except:
        print("video cannot be loaded by decord: ", fname)
        return []

    vr.seek(0)
    frame_id_list = list(range(len(vr)))
    buffer = vr.get_batch(frame_id_list).asnumpy()

    return {
        'video': buffer,
        'frame_id': frame_id_list
    }

# ----- EchoJEPA -----
import warnings
import math
from logging import getLogger
from typing import Optional

logger = getLogger()

def loadvideo_decord_echojepa(
    sample_uri: str,
    fpc: int,  # frames per clip
    frame_step: Optional[int] = None,
    duration: Optional[int] = None,
    fps: Optional[int] = None,
    num_clips: int = 1,
    filter_long_videos: int = int(1e9),  # in bytes, set to a large number to disable
    filter_short_videos: bool = False,
    random_clip_sampling: bool = False,
    allow_clip_overlap: bool = False,
):
    """
    Unified loader:
        - Local path: matches the default filesystem logic exactly.
        - S3 path: mirrors the same semantics (size check, skip behavior, sampling math).
    Returns (buffer[T,H,W,3], clip_indices) or ([], None) on skip/failure.
    """
    fname = sample_uri
    if not os.path.exists(fname):
        warnings.warn(f"video path not found fname='{fname}'")
        return [], None

    _fsize = os.path.getsize(fname)
    if filter_long_videos and _fsize > filter_long_videos:
        warnings.warn(f"skipping long video of size _fsize={_fsize} (bytes)")
        return [], None

    try:
        vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
    except Exception:
        return [], None

    buffer, clip_indices = sample_from_vr(
        vr,
        fpc,
        frame_step,
        duration,
        fps,
        num_clips,
        filter_short_videos,
        random_clip_sampling,
        allow_clip_overlap
    )

    return {
        'video': buffer,
        'frame_id': clip_indices
    }

def sample_from_vr(
    vr: VideoReader,
    fpc: int,
    frame_step: Optional[int] = None,
    duration: Optional[int] = None,
    fps: Optional[int] = None,
    num_clips: int = 1,
    filter_short_videos: bool = False,
    random_clip_sampling: bool = False,
    allow_clip_overlap: bool = False,
):
    fstp = frame_step            
    if duration is not None or fps is not None:
        try:
            video_fps = math.ceil(vr.get_avg_fps())
        except Exception as e:
            logger.warning(e)
            # keep parity with default (no fallback change)
        if duration is not None:
            assert fps is None
            fstp = int(duration * video_fps / fpc)
        else:
            assert duration is None
            fstp = video_fps // fps

    assert fstp is not None and fstp > 0
    clip_len = int(fpc * fstp)

    if filter_short_videos and len(vr) < clip_len:
        warnings.warn(f"skipping video of length {len(vr)}")
        return [], None

    vr.seek(0)  # Go to start of video before sampling frames

    # Partition video into equal sized segments and sample each clip
    partition_len = len(vr) // num_clips

    all_indices, clip_indices = [], []
    for i in range(num_clips):
        if partition_len > clip_len:
            # sample a random window of clip_len frames within the segment
            end_indx = clip_len
            if random_clip_sampling:
                end_indx = np.random.randint(clip_len, partition_len)
            start_indx = end_indx - clip_len
            indices = np.linspace(start_indx, end_indx, num=fpc)
            indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
            indices = indices + i * partition_len
        else:
            if not allow_clip_overlap:
                base = partition_len // fstp
                indices = np.linspace(0, partition_len, num=base)
                if base < fpc:
                    indices = np.concatenate(
                        (indices, np.ones(fpc - base) * partition_len)
                    )
                indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                sample_len = min(clip_len, len(vr)) - 1
                base = sample_len // fstp
                indices = np.linspace(0, sample_len, num=base)
                if base < fpc:
                    indices = np.concatenate(
                        (indices, np.ones(fpc - base) * sample_len)
                    )
                indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                clip_step = 0
                if len(vr) > clip_len and num_clips > 1:
                    clip_step = (len(vr) - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step

        clip_indices.append(indices)
        all_indices.extend(list(indices))

    buffer = vr.get_batch(all_indices).asnumpy()
    return buffer, clip_indices


