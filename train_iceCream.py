# For dataset of iceCream
from argparse import ArgumentParser
import os
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import random
import datetime
import warnings
warnings.filterwarnings("ignore")

from a2vqa import *
import sys



# allow importing utility helpers from parent `models/benchmark` folder
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
try:
    # prefer explicit helpers
    from util import read_MOS_file, compute_metrics
except Exception as e:
    import traceback
    print(f"Failed to import util.py helpers: {e}", file=sys.stderr)
    traceback.print_exc()
    raise ImportError("Could not import util.read_MOS_file; ensure 'models/benchmark' is on PYTHONPATH and required packages (pandas, scipy) are installed") from e



class VQADataset(Dataset):
    def __init__(self, video_features_dir, audio_feature_dir, allindex, dis_names, allmos, scale, features_len):
        super(VQADataset, self).__init__()
        self.allindex = allindex
        self.dis_names = dis_names
        self.video_dir = video_features_dir
        self.audio_dir = audio_feature_dir
        self.allmos = np.array(allmos) / scale
        self.features_len = features_len

    def __len__(self):
        return len(self.allindex)

    def __getitem__(self, index):
        features = np.zeros((self.features_len, 2048, 6, 9)) # 60, 4096
        afeatures = np.zeros((self.features_len, 2048)) #2024 for CNN
          
        tmpvideo = np.load(self.video_dir  + self.dis_names[self.allindex[index]][0] + '.npy')

        tmpaudio = np.load(self.audio_dir +  self.dis_names[self.allindex[index]][1] + '.npy')
        features = tmpvideo[:self.features_len, :, :,:]
        
        afeatures = tmpaudio[:self.features_len, :2048]

        sample = features, afeatures, np.array([self.allmos[self.allindex[index]]])
        return sample





if __name__ == "__main__":
    parser = ArgumentParser(description='"A2VQA: Quality Assessment')
    parser.add_argument("--seed", type=int, default=2025) 
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--model', default='A2VQA', type=str,
                        help='model name (default: A2VQA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument('--trained_model_path', type=str, default='./ckpts',
                        help='path to save model checkpoint')
    parser.add_argument('--database', type=str, default='iceCream',
                        help='database name, iceCream or LIVE_SJTU')

    
    args = parser.parse_args()


    trained_model_path = args.trained_model_path
    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)


    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    print(args)
    print("========================================")

    database = args.database  # 'iceCream' or 'LIVE_SJTU'
    if database == 'iceCream':
        videofeatures_dir = 'video_features_all/A2VQA/iceCream/' 
        audiofeatures_dir = 'audio_features_all/A2VQA/iceCream/'
        Info, _ = read_MOS_file(database, av=2)
        disNames = []
        Mos = []
        for i in range(len(Info['video_name'])):
            video_name = Info['video_name'][i]
            disNames.append([video_name + '.mp4', video_name + '.wav'])
            Mos.append(float(Info['score'][i]))


    print('EXP ID: {}'.format(args.exp_id))
    print(args.model)

    time_str = datetime.datetime.now().strftime("%I%M%B%d")


    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    print("\n\n===========================================================")
    print("===========================================================")
    print('Training on iceCream database...')

    all_SRCC = []
    all_KROCC = []
    all_PLCC = []
    all_RMSE = []
    for i_all in range(10):
        print("========================================")
        print('= = SSplit {}...'.format(i_all))
        index = [i for i in range(len(disNames))]
        test_indexs = []
        split = [0, 1]
        for i_cat in range(10): # 10 categories in the iceCream database
            if i_all < 5:
                test_index_cat = [i for i in range(i_cat * 200 + 40*i_all, i_cat*200 + 40+40*i_all)] # each cat has 200 videossting
            elif i_all < 9:
                test_index_cat = [i for i in range(i_cat * 200 + 20+40*(i_all-5), i_cat*200 + 20+40+40*(i_all-5))]
            else:
                test_index_cat = [i for i in range(i_cat * 200, i_cat*200 + 20)]
                test_index_cat.extend([i for i in range(i_cat * 200 + 180, i_cat*200 + 200)])
            test_indexs.extend(test_index_cat)
        train_indexs = [i for i in index if i not in test_indexs]
        
    

        trained_model_file = trained_model_path + '/A2VQA_' + database + "_SPLIT"  + str(i_all)
        random.seed(i_all+args.seed)
        random.shuffle(train_indexs)
        scale = max(5, max(Mos)) # label normalization factor

        train_dataset = VQADataset(videofeatures_dir, audiofeatures_dir, train_indexs, disNames, Mos, scale=scale, features_len=30)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0)

        val_dataset = VQADataset(videofeatures_dir, audiofeatures_dir, train_indexs, disNames, Mos, scale=scale, features_len=30)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, pin_memory=False, num_workers=0)

        test_dataset = VQADataset(videofeatures_dir, audiofeatures_dir, test_indexs, disNames, Mos, scale=scale, features_len=30)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, pin_memory=False, num_workers=0)
    
        print('Train dataset size: {}, Test dataset size: {}'
            .format(len(train_dataset), len(test_indexs)))

        model = A2VQA(feature_dim=512, hidden_dim=128, hidden_dim2=256).to(device)
        print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))
        model = model.to(device)  

        criterion = nn.MSELoss() 
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_criterion = 100 
        for epoch in range(args.epochs):
            # Train
            model.train()
            L = 0
            for i, (features, afeatures, label) in enumerate(train_loader):
                features = features.to(device).float()
                afeatures = afeatures.to(device).float()
                label = label.to(device).float()
                optimizer.zero_grad()  
                outputs = model(features, afeatures)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (i + 1)
            print('{}-{}:train loss {}'.format(i_all, epoch, train_loss))



            model.eval()
            # Val
            y_pred = np.zeros(len(train_indexs))
            y_val = np.zeros(len(train_indexs))
            L = 0
            with torch.no_grad():
                for i, (features, afeatures, label) in enumerate(val_loader):
                    y_val[i] = scale * label.item()  #
                    features = features.to(device).float()
                    afeatures = afeatures.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features, afeatures)
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            val_loss = L / (i + 1)
            [val_SROCC, val_KROCC, val_PLCC, val_RMSE] = compute_metrics(y_pred, y_val)
        

            print("Val results: {}-{}, val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                .format(i_all, epoch, val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))

            # Update the model with the best val_RMSE
            if val_RMSE < best_val_criterion:
                torch.save(model.state_dict(), trained_model_file)
                print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
                
                # test
                y_pred = np.zeros(len(test_indexs))
                y_test = np.zeros(len(test_indexs))
                L = 0
                with torch.no_grad():
                    for i, (features, afeatures, label) in enumerate(test_loader):
                        y_test[i] = scale * label.item()  #
                        features = features.to(device).float()
                        afeatures = afeatures.to(device).float()
                        label = label.to(device).float()
                        outputs = model(features, afeatures)
                        y_pred[i] = scale * outputs.item()
                        loss = criterion(outputs, label)
                        L = L + loss.item()
                test_loss = L / (i + 1)
                [SROCC, KROCC, PLCC, RMSE] = compute_metrics(y_pred, y_test)


                best_val_criterion = val_RMSE  
                best_SRCC = SROCC
                best_PLCC = PLCC
                best_KROCC = KROCC
                best_RMSE = RMSE
                print("== Test results: {}-{}, test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                    .format(i_all, epoch, test_loss, SROCC, KROCC, PLCC, RMSE))

        all_SRCC.append(best_SRCC)
        all_KROCC.append(best_KROCC)
        all_PLCC.append(best_PLCC)
        all_RMSE.append(best_RMSE)


    print("all SRCC: ", all_SRCC)
    print("all PLCC: ", all_PLCC)
    print("all KROCC: ", all_KROCC)
    print("all RMSE: ", all_RMSE)
    print("Average results over 10 runs:")
    print("Average SRCC: {:.4f}, Average PLCC: {:.4f}, Average KROCC: {:.4f}, Average RMSE: {:.4f}"
        .format(np.mean(all_SRCC), np.mean(all_PLCC), np.mean(all_KROCC), np.mean(all_RMSE)))
    print("Standard deviation of SRCC: {:.4f}, PLCC: {:.4f}, KROCC: {:.4f}, RMSE: {:.4f}"
        .format(np.std(all_SRCC), np.std(all_PLCC), np.std(all_KROCC), np.std(all_RMSE)))


