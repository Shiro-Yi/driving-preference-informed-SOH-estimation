import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Please revised file name when run the code on the corresponding device
device_id = 'D02'
sequence_features_file_dir = './data/features/sequence/D02_device_standardized_sequence_features.csv'
point_features_folder_dir = './data/features/point/D02_device_standardized_point_features'
labels_file_dir = './data/labels/D02_device_reconstructed_SOH_references.csv'
driving_preference_features_file_path = './data/features/standardized_driving_preference_features.csv'
prior_weights_file_dir = './data/features/prior_weights.csv'
model_output_file_path = './results/test_output/D02_model_output.csv'

## Dataset function
class SOH_Features_Dataset(Dataset):
    def __init__(self, device, sequence_features_dir, point_features_dir, labels_dir, driving_preference_features_dir, prior_weights_dir):
        self.device_id = device
        self.sequence_features_file_dir = sequence_features_dir
        self.point_features_folder_dir = point_features_dir
        self.labels_file_dir = labels_dir
        self.driving_preference_features_file_dir = driving_preference_features_dir
        self.prior_weights_file_dir = prior_weights_dir

        self.sequence_features = []
        self.point_features = []
        self.labels = []
        self.weights = []
        self.preference_features = []

        preference_features_df = pd.read_csv(self.driving_preference_features_file_dir)
        prior_weights_df = pd.read_csv(self.prior_weights_file_dir)

        point_features_files_list = os.listdir(self.point_features_folder_dir)
        point_features_files_list.sort(key = lambda x: int(x.split('_')[0]))
        point_features_files = [f for f in point_features_files_list if f.endswith('.csv')]

        for point_features_file in point_features_files:
            segment_point_features_file_path = os.path.join(self.point_features_folder_dir, point_features_file)
            segment_point_feature_df = pd.read_csv(segment_point_features_file_path)
            segment_point_feature = np.array(segment_point_feature_df)
            self.point_features.append(segment_point_feature)

        sequence_features_df = pd.read_csv(self.sequence_features_file_dir)
        segment_sequence_features = np.array(sequence_features_df.drop(columns=['segment']))
        self.sequence_features.extend(segment_sequence_features)

        labels_df = pd.read_csv(self.labels_file_dir)
        device_labels = np.array(labels_df.drop(columns=['segment']))
        self.labels.extend(device_labels)

        device_prior_weights_df = prior_weights_df[prior_weights_df['device'] == self.device_id]
        device_prior_weights = np.array(device_prior_weights_df.drop(columns=['device'])).reshape(1,-1)
        self.weights.extend(np.tile(device_prior_weights,(labels_df.shape[0],1)))

        device_preference_features_df = preference_features_df[preference_features_df['device']==device_id]
        device_preference_features = np.array(device_preference_features_df.iloc[:,1:]).reshape(1,-1)
        device_preference_features = np.tile(device_preference_features,(labels_df.shape[0],1))
        self.preference_features.extend(device_preference_features)

        self.sequence_features = np.array(self.sequence_features).reshape(-1,10)
        self.point_features = np.array(self.point_features)
        self.labels = np.array(self.labels).reshape(-1,1)
        self.weights = np.array(self.weights).reshape(-1,4)
        self.preference_features = np.array(self.preference_features).reshape(-1,2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        segment_sequence_features = torch.from_numpy(self.sequence_features[idx])
        segment_point_features = torch.from_numpy(self.point_features[idx]).reshape(8,20)
        segment_C_rate_features = segment_point_features[0:2,:]
        segment_T_features = segment_point_features[2:6,:]
        segment_SOC_features = segment_point_features[6:8,:]
        segment_label = torch.from_numpy(self.labels[idx])
        segment_weights = torch.from_numpy(self.weights[idx])
        segment_preference_features = torch.from_numpy(self.preference_features[idx])

        return segment_sequence_features, segment_C_rate_features, segment_T_features, segment_SOC_features, segment_label, segment_weights, segment_preference_features

## Sub-models construction
class Sub_Model(nn.Module):
    def __init__(self):
        super(Sub_Model, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        self.CNN_C_rate = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=7, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, padding=0, stride=1),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=0, stride=1),
            nn.Flatten(),
        )

        self.CNN_T = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, padding=0, stride=1),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=0, stride=1),
            nn.Flatten(),
        )

        self.CNN_SOC = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=0, stride=1),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=5, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=0, stride=1),
            nn.Flatten(),
        )

        self.FC = nn.Linear(44, 1)

        self.dropout_layer_mlp = nn.Dropout(p=0.1)
        self.dropout_layer_cnn = nn.Dropout(p=0.3)

    def forward(self, sequence_features, C_rate_point_features, T_point_features, SOC_point_features):
        MLP_features = self.dropout_layer_mlp(self.MLP(sequence_features))
        CNN_C_rate_features = self.dropout_layer_cnn(self.CNN_C_rate(C_rate_point_features))
        CNN_T_features = self.dropout_layer_cnn(self.CNN_T(T_point_features))
        CNN_SOC_features = self.dropout_layer_cnn(self.CNN_SOC(SOC_point_features))
        connected_features = torch.cat([MLP_features,CNN_C_rate_features,CNN_T_features,CNN_SOC_features], dim=1)
        prediction = self.FC(connected_features)
        return prediction

## Sub-models integration
class Ensemble_Frame(nn.Module):
    def __init__(self):
        super(Ensemble_Frame, self).__init__()

        self.Moderate_Model = Sub_Model()
        self.Normal_Model = Sub_Model()
        self.Aggressive_Model = Sub_Model()
        self.Extreme_Model = Sub_Model()

        self.gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def get_alpha(self, preference_features):
        with torch.no_grad():
            alpha = self.gate(preference_features)
            alpha = alpha.cpu().detach().numpy()
        return alpha

    def forward(self, sequence_features, C_rate_point_features, T_point_features, SOC_point_features, sample_weights, preference_features):
        Moderate_pre = self.Moderate_Model(sequence_features, C_rate_point_features, T_point_features, SOC_point_features)
        Normal_pre = self.Normal_Model(sequence_features, C_rate_point_features, T_point_features, SOC_point_features)
        Aggressive_pre = self.Aggressive_Model(sequence_features, C_rate_point_features, T_point_features, SOC_point_features)
        Extreme_pre = self.Extreme_Model(sequence_features, C_rate_point_features, T_point_features, SOC_point_features)

        alpha = self.gate(preference_features)
        weights = 0.25*(1-alpha) + sample_weights * alpha

        sub_model_preds = torch.cat([Moderate_pre,Aggressive_pre,Normal_pre,Extreme_pre], dim=1)
        estimation = torch.sum(sub_model_preds*weights, dim=1, keepdim=True)

        return estimation

## Load model weights
model_path = './model/driving_preference_informed_multi_models_framework.pth'
model = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
model.eval()

## Load features and labels for test device
features_dataset = SOH_Features_Dataset(device=device_id,
                                        sequence_features_dir=sequence_features_file_dir,
                                        point_features_dir=point_features_folder_dir,
                                        labels_dir=labels_file_dir,
                                        driving_preference_features_dir=driving_preference_features_file_path,
                                        prior_weights_dir=prior_weights_file_dir)

test_loader = DataLoader(features_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=0)

model_output = []

## Acquire model outputs and save as csv files
with torch.no_grad():
    for batch_idx, (sequence_features, C_rate_point_features, T_point_features, SOC_point_features, labels, weights, preference_features) in enumerate(test_loader):
        sequence_features = sequence_features.cuda()
        sequence_features = sequence_features.type(torch.cuda.FloatTensor)
        C_rate_point_features = C_rate_point_features.cuda()
        C_rate_point_features = C_rate_point_features.type(torch.cuda.FloatTensor)
        T_point_features = T_point_features.cuda()
        T_point_features = T_point_features.type(torch.cuda.FloatTensor)
        SOC_point_features = SOC_point_features.cuda()
        SOC_point_features = SOC_point_features.type(torch.cuda.FloatTensor)
        labels = labels.cuda()
        labels = labels.type(torch.cuda.FloatTensor)
        weights = weights.cuda()
        weights = weights.type(torch.cuda.FloatTensor)
        preference_features = preference_features.cuda()
        preference_features = preference_features.type(torch.cuda.FloatTensor)

        outputs = model(sequence_features, C_rate_point_features, T_point_features, SOC_point_features, weights, preference_features)
        model_output.append(outputs.squeeze().cpu().numpy())

model_output = np.array(model_output)
model_output_df = pd.DataFrame(model_output,columns=['Model Estimations'])
labels_df = pd.read_csv(labels_file_dir)
model_output_df = pd.concat([labels_df,model_output_df],axis=1)
model_output_df.to_csv(model_output_file_path, index=False)

print(f"Device {device_id} Test Complete.")