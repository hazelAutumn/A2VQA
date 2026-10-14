# Fixed number of frames sampled
# The file used to extract feature maps for each video for ResNet30maps
# UPDATED DATE: 13.03.2026
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo

import skvideo.io
from PIL import Image
import os
import numpy as np
import random
from argparse import ArgumentParser
import scipy
import torch.nn.functional as functions


import sys
# allow importing utility helpers from parent `models/benchmark` folder
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
try:
    # prefer explicit helpers
    from util import read_MOS_file
except Exception:
    # fall back: keep script functioning even when util is unavailable
   read_MOS_file = None

            

class VideoDatasetMutipleResolution(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, videos_dir, video_names, video_format='RGB', widths=None, heights=None):
        super(VideoDatasetMutipleResolution, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.format = video_format
        self.widths = widths
        self.heights = heights

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        if isinstance(self.widths, list) and isinstance(self.heights, list):
            video_height = self.heights[idx]
            video_width = self.widths[idx]
        else:
            video_height = self.heights
            video_width = self.widths          

        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), video_width, video_height,
                                      inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]


        transformed_dis_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            dis_frame = video_data[frame_idx]
            dis_frame = Image.fromarray(dis_frame)
            dis_frame = transform(dis_frame)
            transformed_dis_video[frame_idx] = dis_frame


        sample = {'video_name':video_name, 'dis_video': transformed_dis_video}
        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                feat_interpolate = functions.interpolate(x, size=(6,9), mode='bilinear')
                return feat_interpolate
            



def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)

# Only process the whole frames, not using patches
# this is for in the wild file with different resolutions
def get_features_multi(video_data, frame_num = 30, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)

    dis_output = torch.Tensor().to(device)
    extractor.eval()
    frame_length = video_data.shape[0]
    frame_interval = int(frame_length/frame_num)
    icount = 0
    with torch.no_grad():
        for iframe in range(0, frame_length, frame_interval):
            if icount + 1 > frame_num:
                break
            dis_batch = video_data[iframe:iframe + 1, 0:3,:,:].to(device)
            feat_maps = extractor(dis_batch)
            dis_output = torch.cat((dis_output, feat_maps), 0)
            icount = icount + 1
        dis_output = dis_output.squeeze()
    return dis_output





if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument('--dataset', default='iceCream', type=str,
                        help='database name') # iceCream or LIVE_SJTU
    parser.add_argument('--frame_num', default=30, type=int,
                        help='Total number of sampled frames')
    parser.add_argument('--frame_batch_size', type=int, default=5,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    features_dir_main = '/home/user1/avqa/models/benchmark/video_features_all/'
    if not os.path.exists(features_dir_main):
        os.makedirs(features_dir_main)
    features_dir = os.path.join(features_dir_main, 'A2VQA')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    database = args.dataset
    print("Database: ", database)
    print("=================================================")
    if database == 'LIVE_SJTU':
        dis_videos_dir = '/home/user1/LIVE_STJU/LIVE-SJTU_AVQA/Distorted/'
        features_dir = 'videos/LIVE_SJTU/'
        datainfo = '/home/user1/LIVE_STJU/LIVE-SJTU_AVQA/MOS.mat'

    
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    disable_gpu = False
    device = torch.device("cuda" if disable_gpu and torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    dis_video_names = []


    if database == 'iceCream':
        print("Processing iceCream dataset")
        features_dir = os.path.join(features_dir, 'iceCream/')
        Info, videos_dir = read_MOS_file(database, av=1)
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)

        video_names = []
        widths = []
        heights = []
        for i in range(len(Info)):
            if 'lll2' in Info[i]:
               width = 640
               height = 360
            elif 'lll3' in Info[i]:
               width = 256
               height = 256
            else:
               width = 1920
               height = 1080
            widths.append(width)
            heights.append(height)
            vName = Info[i] + '.mp4'
            video_names.append(vName)
        dis_dataset = VideoDatasetMutipleResolution(videos_dir, video_names, 'RGB', widths=widths, heights=heights)
        print("Number of videos: ", len(dis_dataset))
        for i in range(len(dis_dataset)): 
            current_data = dis_dataset[i]
            current_dis_video = current_data['dis_video']
            height = heights[i]
            width = widths[i]
            #print('height: ', height)


            dis_features= get_features_multi(current_dis_video, args.frame_num, device)

            save_path = os.path.join(features_dir,  current_data['video_name'] + '.npy')
            dis_features = dis_features.to('cpu').numpy()
            np.save(save_path, dis_features)

    elif database == 'LIVE_SJTU':
        Info = scipy.io.loadmat(datainfo)
        dis_video_names = [''.join(Info['disNames'][i, 0]) for i in range(0, len(Info['disNames']), 3)]
        height = int(Info['resolution'][0][0])
        width = int(Info['resolution'][0][1])

        dis_dataset = VideoDataset(dis_videos_dir, dis_video_names, 'YUV420', height=height, width=width)

        for i in range(0, len(dis_dataset)):
            current_data = dis_dataset[i]
            current_dis_video = current_data['dis_video']

            dis_features = get_features_multi(current_dis_video, args.frame_num, device)
            dis_features = dis_features.to('cpu').numpy()
            save_path = os.path.join(features_dir, current_data['dis_video_name'] + '.npy')
            np.save(save_path, dis_features)

    else:
        print("Database name not recognized. Please check again.")