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
    def __init__(self, path=config["container"]["model_path"]):
        self.device = 'cuda'

        self.embedmodel = imagebind_model.imagebind_huge(pretrained=True)
        self.embedmodel.eval()
        self.embedmodel.to(self.device)

        # Load Classifier
        with open(path, 'rb') as f: 
            self.classmodel = pickle.load(f)

    def tag(self, video_path: str) -> List[VideoTag]:
        label, cls, _ = self._embedPredict(video_path)
        if cls == 0:
            # negative class
            return []
        end_time = int(get_video_duration(video_path) * 1000)  # Convert to milliseconds
        return [VideoTag(text=label, confidence=1.0, start_time=0, end_time=end_time)]
    
    def _embedPredict(self, vidfname, buffer = 5):
        # Load Imagebind (Video Embedding)
        class2label = {0: 'Other', 1: 'Windmill'}
        # Can speed up by batch processing / parallel
        event_inputs = {ModalityType.VISION: data.load_and_transform_video_data([vidfname], self.device, clips_per_video=10),}
            
        all_data = event_inputs['vision']
        maxlen = all_data.shape[1]
        print('Total No. of Frames {}'.format(maxlen))
        
        all_part_labels = [] 
        for i in range(0,maxlen-buffer):
            st = i
            end = min(st+buffer,maxlen)
            
            # print('St {}, End {}'.format(st, end))

            event_inputs['vision'] = all_data[:,st:end]
            # print(event_inputs['vision'].shape)

            with torch.no_grad(): event_embeddings = self.embedmodel(event_inputs)
            tstX = event_embeddings['vision'].cpu().numpy()         # This is the embedding!
            ypred = int(self.classmodel.predict(tstX))
            all_part_labels.append(ypred)
            # print('This is class {} --> {}'.format(ypred, class2label[ypred]))
        
        ypred = 1 if sum(all_part_labels)>=3.0 else 0
        print('Final Prediction --> {}, {}'.format(ypred, class2label[ypred]))
        return class2label[ypred], ypred, all_part_labels