import json
import os
import pickle
import shutil
import subprocess
from typing import List

import numpy as np
import torch
import torch.nn as nn
from common_ml.model import VideoModel
from common_ml.tags import VideoTag
from common_ml.utils.metrics import timeit
from imagebind.data import SpatialCrop, get_clip_timepoints, uniform_crop
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from loguru import logger
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from config import config


def get_video_duration(path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "json", path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = json.loads(result.stdout)['format']['duration']
    return float(duration)

class WindmillDetector(VideoModel):
    def __init__(self, path=config["container"]["model_path"]):
        self.device = 'cuda'

        self.embedmodel = imagebind_model.imagebind_huge(pretrained=True)
        self.embedmodel.eval()
        self.embedmodel.to(self.device)

        # Load Classifier
        with open(path, 'rb') as f:
            self.classmodel = pickle.load(f)

    def tag(self, video_path: str) -> List[VideoTag]:
        labelmap = {0:'None', 1: 'Windmill'}
        with timeit("Getting segs"):
            vid_pix_i, vid_segs_in_sec = self.getvidsegs(video_path,
                    clip_duration = 2,
                    clips_per_video =20,
                    stride = 5, device = torch.device('cuda'),
                    verbose = 1)
        if vid_pix_i is None:
            return []
        vid_pix_i = vid_pix_i.to('cuda')
        event_inputs = {ModalityType.VISION: vid_pix_i,}
        with timeit("Getting Embedding"):
            with torch.no_grad():
                event_embeddings = self.embedmodel(event_inputs)
        embed_np = event_embeddings['vision'].cpu().numpy()
        logger.info('The Embedding {}'.format(embed_np.shape))
        with timeit("Predicting"):
            ypred = self.classmodel.predict(embed_np)
        labelpred = [labelmap[y] for y in ypred]
        logger.info('The Label Prediction {}'.format(labelpred))

        vid_end = int(get_video_duration(video_path) * 1000)  # Convert to milliseconds

        res = []
        for pred, seg in zip(labelpred, vid_segs_in_sec):
            if pred == 'None':
                continue
            start_time = int(seg[0] * 1000)
            end_time = int(seg[1] * 1000)
            if start_time > vid_end:
                start_time = vid_end
                logger.warning(f"Start time {end_time} exceeds video duration, setting to {vid_end}")
            if end_time > vid_end:
                end_time = vid_end
                logger.warning(f"End time {end_time} exceeds video duration, setting to {vid_end}")
            res.append(VideoTag(text=pred, confidence=1.0, start_time=start_time, end_time=end_time))

        return res

    def getvidsegs(self,
                   video_path,
                    clip_duration = 2,
                    clips_per_video =20,
                    stride = 5, device = torch.device('cuda'),
                    verbose = 0): # Assume one vid has one label
        '''This function creates input tensors for video segments of duration 5s.'''


        video_outputs = []
        vid_ts = []
        # Get the Video
        video = EncodedVideo.from_path(
                    video_path,
                    decoder="decord",
                    decode_audio=False,
                    **{"sample_rate": 16000},
                )

        # Tranformation as per imagebind source code!
        video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(224),
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        dur_full = video.duration
        dur = int(dur_full)

        if verbose>1: logger.info('The total Clip Duration ={}'.format(dur_full))

        for st in range(0, dur, stride):
            end = min(st+stride,dur)
            vid_ts.append([st,end])
            if verbose>0: logger.info('Processing chunk [{}, {}] from {}'.format(st, end, dur))
            clip = video.get_clip(st, end)
            clip_seg_dur = end-st

            if clip_seg_dur<=clip_duration: clip_duration_var = 0.5*clip_seg_dur
            else: clip_duration_var =clip_duration

            if clip is None: raise ValueError("No clip found")

            clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration_var, clips_per_video=clips_per_video)
            frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)
            all_clips_timepoints = get_clip_timepoints(clip_sampler, clip_seg_dur)

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                # Read the clip, get frames
                st_pnt = st+clip_timepoints[0]
                ed_pnt = st+clip_timepoints[1]

                if verbose>1: logger.info(st_pnt, ed_pnt)
                clip = video.get_clip(st_pnt,ed_pnt)

                if clip is None: raise ValueError("No clip found")
                video_clip = frame_sampler(clip["video"])
                video_clip = video_clip / 255.0  # since this is float, need 0-1
                # logger.info(video_clip.shape)
                all_video.append(video_clip)

            all_video = [video_transform(clip) for clip in all_video]
            all_video = SpatialCrop(224, num_crops=3)(all_video)
            # logger.info('After Video Transform')
            # logger.info('Len = {}, shape = {}'.format(len(all_video), all_video[0].shape))

            all_video = torch.stack(all_video, dim=0).to('cpu')
            video_outputs.append(all_video)

        if len(video_outputs)>0: result = torch.stack(video_outputs, dim=0).to('cpu')
        else: result = None
        return result, vid_ts