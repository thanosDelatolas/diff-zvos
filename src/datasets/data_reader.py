import os
from omegaconf import DictConfig
from datasets.video_reader import VideoReader
from datasets.video_reader_per_object import VideoReaderPerObject
from datasets.video_reader_oracle import VideoReaderOracle

class DatasetReader:
    def __init__(self, data_root: str, dataset_cfg: DictConfig, video_reader_type: str, device):
        self.data_root = data_root
        self.dataset_cfg = dataset_cfg
        self.image_directory = os.path.join(data_root, dataset_cfg['image_directory'])
        self.mask_directory = os.path.join(data_root, dataset_cfg['mask_directory'])

        self.is_mose = 'MOSE' in dataset_cfg['image_directory']
        subset = dataset_cfg.get('subset', None)
        if subset:
            with open(os.path.join(data_root, subset)) as f:
                self.vid_list = sorted([line.strip() for line in f])
        else :
            self.vid_list = sorted(os.listdir(self.mask_directory))

        self.idx = 0
        self.max_idx = len(self.vid_list)
        self.device = device
        self.video_reader_type = video_reader_type
        assert video_reader_type in ['per_object', 'default', 'oracle']
        """
        per_object:
            iterates over each object in the video
        default:
            iterates over all objects simultaneously
        oracle:
            iterates over each object in the video and returns its gt mask for each frame
        """

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        
        vid_name = self.vid_list[self.idx]
        curr_image_directory = os.path.join(self.image_directory, vid_name)
        curr_mask_directory = os.path.join(self.mask_directory, vid_name)
        self.idx += 1

        if self.video_reader_type == 'per_object':
            video_reader = VideoReaderPerObject(curr_image_directory, curr_mask_directory, vid_name, self.is_mose, self.device)
        elif self.video_reader_type == 'default':
            video_reader = VideoReader(curr_image_directory, curr_mask_directory, vid_name, self.is_mose, self.device)
        elif self.video_reader_type == 'oracle':
            video_reader = VideoReaderOracle(curr_image_directory, curr_mask_directory, vid_name, self.device)
        return video_reader
        

    def __len__(self):
        return len(self.vid_list)
