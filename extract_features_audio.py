"""
The final version, using melspectrogram to extract audio features
"""

import torch
import torchaudio
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from argparse import ArgumentParser
import random
import scipy.io


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

def convertToMelSpectrogram(audio_path, n_fft=512, hop_length=256):
    """Convert audio to Mel spectrogram"""
    y, sr = librosa.load(audio_path, sr=None)
    n_mels = 224
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    tStamp = np.empty((S.shape[1]),dtype=np.float64)
    for i in range(len(tStamp)):
        tStamp[i] = (i * hop_length + n_fft//2) / sr
    return S_db, tStamp


class AudioDatasetCustom(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, audios_dir, audios_names, audio_frameRate):
        super(AudioDatasetCustom, self).__init__()
        self.audios_dir = audios_dir
        self.audios_names = audios_names
        self.audios_frameRate = audio_frameRate

    def __len__(self):
        return len(self.audios_names)

    def __getitem__(self, idx):
        audios_names = self.audios_dir + self.audios_names[idx]
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        [S, T] = convertToMelSpectrogram(audios_names)


        transforms_audio = transform(S)
        sample = {'audio_name':self.audios_names[idx], 'audio': transforms_audio, 'tStamp': T, 'frameRate': self.audios_frameRate[idx]}

        return sample
    
def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)
class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                feature_std = global_std_pool2d(x)
                return features_mean, feature_std

def get_features(audios_data, audio_tStamp, frameRate,
                 device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)

    extractor.eval()

    patchSize = 224
    frameSkip = int(frameRate*8/30) #sample 30 frames from 8 seconds
    with torch.no_grad():
        for iFrame in range(1, int(frameRate*8),frameSkip):
            tCenter =np.argmin(abs(audio_tStamp - iFrame / frameRate))
            tStart = tCenter - patchSize / 2 + 1
            tEnd = tCenter + patchSize / 2
            if tStart < 1:
                tStart = 1
                tEnd = patchSize
            else:
                if tEnd > audios_data.shape[2]:
                    tStart = audios_data.shape[2] - patchSize + 1
                    tEnd = audios_data.shape[2]
            specRef_patch = audios_data[:, :, int(tStart-1): int(tEnd)]
            refRGB = torch.cat((specRef_patch, specRef_patch, specRef_patch),0)

            last_batch = refRGB.view(1,3,specRef_patch.shape[1],specRef_patch.shape[2]).float().to(device)
            features_mean, feature_std = extractor(last_batch)
            #print("features_mean shape: ", features_mean.shape)

            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, feature_std), 0)
        output = torch.cat((output1, output2), dim=1)
        output = output.squeeze()
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument('--dataset', default='iceCream', type=str,
                        help='database name (default: iceCream)') # LIVE_SJTU, iceCream
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    

    torch.utils.backcompat.broadcast_warning.enabled = True
    disable_gpu = False
    device = torch.device("cuda" if disable_gpu and torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    database = args.dataset
    print("Using database: ", database)

    features_dir_main = '/home/user1/audio_features_all/' # your directory
    if not os.path.exists(features_dir_main):
        os.makedirs(features_dir_main)
    features_dir = os.path.join(features_dir_main, 'A2VQA')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    features_dir = os.path.join(features_dir, database)

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    if database == 'iceCream':
        print("Extracting audio features for iceCream dataset...")
        Info, dis_audios_dir = read_MOS_file(database, extra= True, av=0)
        dis_audios_dir = dis_audios_dir + "/wav/"
    elif database == 'LIVE_SJTU':
        dis_audios_dir = '/home/user1/LIVE_STJU/LIVE-SJTU_AVQA/Distorted/'
        features_dir = 'audio_features/LIVE_SJTU_Mel_30f/' # with Mel spectrogram
    else:
        print("Unsupported database: ", database)
    
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    ref_audios_names, dis_audios_names = [], []
    ref_frameRate, dis_frameRate = [], []
    
    
    if database == 'iceCream':
        dis_frameRate = Info['fps']
        print("key s in Info: ", Info.keys())
        for i in range (len(Info)):
            dis_audios_names.append(''.join(Info['audio'][i] + '.wav'))
    
    elif database == 'LIVE_SJTU':
        datainfo = '/home/user1/LIVE_STJU/LIVE-SJTU_AVQA/MOS.mat'
        Info = scipy.io.loadmat(datainfo)
        ref_audios_names = [''.join(Info['refNames'][i, 1]) for i in range(0, len(Info['refNames']), 24)]
        ref_frameRate = [Info['frameRate'][i] for i in range(0, len(Info['frameRate']), 24)]
        for i in range(0,len(Info['disNames']),24):
            dis_audios_names.append(''.join(Info['disNames'][i, 1]))
            dis_audios_names.append(''.join(Info['disNames'][i+1, 1]))
            dis_audios_names.append(''.join(Info['disNames'][i+2, 1]))
            dis_frameRate.append(Info['frameRate'][i])
            dis_frameRate.append(Info['frameRate'][i+1])
            dis_frameRate.append(Info['frameRate'][i+2])
    
    else:
        print("Unsupported database for feature extraction: ", database)
        print("==============")
