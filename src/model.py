from typing import List
import pickle
import subprocess
import json
import shutil

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from common_ml.tags import VideoTag
from common_ml.model import VideoModel

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
    def __init__(self, path=config["storage"]["model_path"]):
        self.path = path

    def tag(self, video_path: str) -> List[VideoTag]:
        label, _ = self._embedPredict(video_path)
        end_time = int(get_video_duration(video_path) * 1000)  # Convert to milliseconds
        return [VideoTag(text=label, confidence=1.0, start_time=0, end_time=end_time)]

    def _embedPredict(self, vidfname, device=torch.device('cpu')):
        # Load Imagebind (Video Embedding)
        class2label = {0: 'Other', 1: 'Windmill'}
        device = 'cuda'
        embedmodel = imagebind_model.imagebind_huge(pretrained=True)

        embedmodel.eval()
        embedmodel.to(device)

        # Load Classifier
        with open(self.path, 'rb') as f: 
            classmodel = pickle.load(f)

        # Can speed up by batch processing / parallel
        event_inputs = {ModalityType.VISION: data.load_and_transform_video_data([vidfname], device),}
        with torch.no_grad(): 
            event_embeddings = embedmodel(event_inputs)
        tstX = event_embeddings['vision'].cpu().numpy()         # This is the embedding!
        ypred = int(classmodel.predict(tstX))

        print('This is class {} --> {}'.format(ypred, class2label[ypred]))
        return class2label[ypred], ypred