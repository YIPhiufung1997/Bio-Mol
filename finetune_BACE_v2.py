
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torchvision.transforms import *
import random 
from torch_geometric.data import Data

# print(device)
import numpy as np
import pandas as pd
# import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import rdkit as rd
from rdkit import Chem
import torch.nn as nn
import h5py 
#from tensorflow import keras
from sklearn import metrics 

import itertools
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv ,SAGEConv
def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(0)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

from torch_geometric.data import Data
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# E4_start=3202-128-128-1
# E4_end=3202-128-1

BACE_data=pd.read_csv("extend_dataset/BACE/BACE_train_4over4.csv")

# BACE_data=BACE_data[~BACE_data["A1.0"].isna()]
BACE_ourimpute_train=BACE_data.iloc[:,0:3200].fillna(0.0).values
BACE_ourimpute_train_y=BACE_data.loc[:,'Class'].values


BACE_data_valid=pd.read_csv("extend_dataset/BACE/BACE_valid_4over4.csv")

# BACE_data=BACE_data[~BACE_data["A1.0"].isna()]
BACE_ourimpute_valid=BACE_data_valid.iloc[:,0:3200].fillna(0.0).values
BACE_ourimpute_valid_y=BACE_data_valid.loc[:,'Class'].values



BACE_data_test=pd.read_csv("extend_dataset/BACE/BACE_test.csv")

# BACE_data=BACE_data[~BACE_data["A1.0"].isna()]
BACE_ourimpute_test=BACE_data_test.iloc[:,0:3200].fillna(0.0).values
BACE_ourimpute_test_y=BACE_data_test.loc[:,'Class'].values


# y
# y_nonan=y[~np.isnan(y)]
# y_nonan=y_nonan.values
# BACE_A1_ourmethod=BACE_A1
# BACE_A1_ourmethod=BACE_A1_ourmethod

# print(BACE_A1_ourmethod.shape)

class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2400),
            torch.nn.ReLU(),
            torch.nn.Linear(2400, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 1600),
            torch.nn.ReLU(),
            torch.nn.Linear(1600, 1400),
            torch.nn.ReLU(),
            torch.nn.Linear(1400, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
#             torch.nn.ReLU(),
#             torch.nn.Linear(36, 18),
#             torch.nn.ReLU(),
#             torch.nn.Linear(18, 9)
        )
        self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(9, 18),
#             torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 800),
            torch.nn.ReLU(),
            torch.nn.Linear(800, 1600),
            torch.nn.ReLU(),
            torch.nn.Linear(1600, input_size),
            
        )
        self.fine_tune= torch.nn.Sequential(
        )

        # self.linear = nn.Linear(8, 2)
        # self.conv3 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        # self.conv4 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning
        

    def forward(self, data):
        encoded=self.encoder(data)
        decoded_1=self.decoder(encoded)
        encoded_1=self.fine_tune(encoded)

        
        
 

        return encoded_1, decoded_1
    
    
    
model=Encoder(3200)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

import os
PATH="encoder_model_v2.pt"
check_file = os.path.isfile(PATH)
if check_file==True:
    print(PATH)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']




model.fine_tune.fc1 = nn.Linear(128, 128) # 定义一个新的FC层
# model.fine_tune.rl1 = nn.ReLU()
model.fine_tune.fc2 = nn.Linear(128, 64) # 定义一个新的FC层
# model.fine_tune.rl1 = nn.ReLU()
model.fine_tune.fc3 = nn.Linear(64, 32) # 定义一个新的FC层
# model.fine_tune.rl2 = nn.ReLU()
model.fine_tune.fc4 = nn.Linear(32, 16) # 定义一个新的FC层

# model.fine_tune.rl3 = nn.ReLU()
model.fine_tune.fc5 = nn.Linear(16, 2) # 定义一个新的FC层
# # # model.fine_tune.rl4 = nn.Sigmoid()
# model.fine_tune.fc5 = nn.Linear(2, 2) # 定义一个新的FC层
# # # model.fine_tune.rl5 = nn.Sigmoid()
# model.fine_tune.fc6 = nn.Linear(2, 2) # 定义一个新的FC层

# model_pretrain.fine_tune.rl2 = nn.ReLU()
model.fine_tune.softmax = nn.Softmax(dim=1)
    



model=model.to(device)
# 定义一个新的FC层model_1=model_1.to(device)# 放到设备中
print(model) # 最后再打印一下新的模型
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)


# from sklearn.model_selection import train_test_split
# # i.e. 70 % training dataset and 30 % test datasets
# BACE_ourimpute_train, BACE_ourimpute_test, BACE_ourimpute_train_y, BACE_ourimpute_test_y = train_test_split(BACE_A1_ourmethod, y_nonan, test_size = 0.30, random_state=42,stratify=y_nonan)
class BACE_Dataset(Dataset):
    def __init__(self, X,y):
#         self.__device = device
#         self.__clazz = clazz
        self.__X = X
        self.Y = y
        # self._neighbor = neighbor_list
        # self.train_ref=train_set
    def __len__(self):
        return self.__X.shape[0]

    def __getitem__(self, idx):
        # anchor_valid=np.transpose(self.__X[idx])
        # anchor_valid = np.reshape(anchor_valid,(1, anchor_valid.size))
        # # print(anchor_valid.shape)

        # get_neighbors=self._neighbor.iloc[idx,0:5]
        # train_ref=self.train_ref.iloc[get_neighbors].values
        # print(train_ref.shape)

        # anchor_sample_concat = np.concatenate((anchor_valid, train_ref), axis=0)
        # print(anchor_sample_concat.shape)

        

        anchor=torch.tensor(self.__X[idx], dtype=torch.float32)
        # anchor=torch.reshape(anchor.t(), (25, 128))

        # nn1=torch.tensor(self.__X.iloc[get_neighbors[1],:])
        # nn1=torch.reshape(nn1.t(), (25, 128))



        # anchor = Data(x=anchor, edge_index=edge_index.t().contiguous())
        label = self.Y[idx]
        
        # print(label)
        

        return anchor, label
    
    ### one hot tranform
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef



# # 

# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=0)
# resample_noimpute_train, resample_noimpute_train_y = rus.fit_resample(BACE_ourimpute_train, BACE_ourimpute_train_y)
inhibitor = [0, 1]



enc = preprocessing.OneHotEncoder(categories=[inhibitor])
enc.fit(BACE_ourimpute_train_y.reshape(-1, 1))
BACE_ourimpute_train_y_onehot=enc.transform(BACE_ourimpute_train_y.reshape(-1, 1)).toarray()
BACE_ourimpute_train_y_onehot=torch.tensor(BACE_ourimpute_train_y_onehot , dtype=torch.float32)

# # print(resample_noimpute_train.shape)

# enc = preprocessing.OneHotEncoder(categories=[inhibitor])
# enc.fit(resample_noimpute_train_y.reshape(-1,1))
# # print(resample_noimpute_train_y)
# resample_noimpute_train_y_onehot=enc.transform(resample_noimpute_train_y.reshape(-1,1)).toarray()
# # print(resample_noimpute_train_y_onehot.shape)
# resample_noimpute_train_y_onehot=torch.tensor(resample_noimpute_train_y_onehot , dtype=torch.float32)

# inhibitor = [0, 1]

enc_val = preprocessing.OneHotEncoder(categories=[inhibitor])
enc_val.fit(BACE_ourimpute_valid_y.reshape(-1, 1))
BACE_ourimpute_valid_y_onehot=enc.transform(BACE_ourimpute_valid_y.reshape(-1, 1)).toarray()
BACE_ourimpute_valid_y_onehot=torch.tensor(BACE_ourimpute_valid_y_onehot , dtype=torch.float32)




# rus_val = RandomUnderSampler(random_state=0)
# resample_noimpute_test, resample_noimpute_test_y = rus_val.fit_resample(BACE_ourimpute_test, BACE_ourimpute_test_y)
enc_test = preprocessing.OneHotEncoder(categories=[inhibitor])
enc_test.fit(BACE_ourimpute_test_y.reshape(-1, 1))
BACE_ourimpute_test_y_onehot=enc.transform(BACE_ourimpute_test_y.reshape(-1, 1)).toarray()
BACE_ourimpute_test_y_onehot=torch.tensor(BACE_ourimpute_test_y_onehot , dtype=torch.float32)



BACE_imputed_ourmethod_ds = BACE_Dataset(X=BACE_ourimpute_train,y=BACE_ourimpute_train_y_onehot)
BACE_imputed_ourmethod_dl = DataLoader(BACE_imputed_ourmethod_ds, batch_size=128, shuffle=True, num_workers=1)






BACE_valid_ds = BACE_Dataset(X=BACE_ourimpute_valid,y=BACE_ourimpute_valid_y_onehot)
sampler = torch.utils.data.RandomSampler(BACE_valid_ds, replacement=False)
BACE_valid_dl = DataLoader(BACE_valid_ds, batch_size=128, shuffle=False, num_workers=1, sampler=sampler)

BACE_test_ds = BACE_Dataset(X=BACE_ourimpute_test,y=BACE_ourimpute_test_y_onehot)
sampler_test = torch.utils.data.RandomSampler(BACE_test_ds, replacement=False)
BACE_test_dl = DataLoader(BACE_test_ds, batch_size=128, shuffle=False, num_workers=1, sampler=sampler_test)


def train_BACE(model,train_loader):
    model.train()
    losses=[]
    for all_data in train_loader:
        train_data_batch, y_batch=all_data
        train_data_batch=train_data_batch.to(device)
        # print(train_data_batch)
        optimizer.zero_grad()
        encode, decode=model(train_data_batch)
        # pred_y=pred_y.unsqueeze(0)

        # print(pred_y)
        y_batch=y_batch.to(device)
        # print(f"expect_y{y_batch}")
        # print(f"predict_encoder{encode}")


        loss_ = criterion(encode, y_batch)
        loss_.backward()
        optimizer.step()
        tmp_loss=loss_.detach().cpu().numpy().item()
        losses.append(tmp_loss)

    losses = np.array(losses)
    print(f'epoch:{epoch} '+
            f'loss:{losses.mean()} ')

    
def valid_BACE(model, valid_dl):
    df = []
    model.eval()
    ourmethod_all=np.empty([1,2])
    y_all=np.empty([1,2])
    for all_data in valid_dl:
        valid_data_batch, y_batch=all_data

        # print(train_data_batch.x)
        optimizer.zero_grad()
        valid_data_batch=valid_data_batch.to(device)
        our_encode,our_decode=model(valid_data_batch)
        
        

        # y_batch_=y_batch.to(device)
#         print(x.shape)
        
        # y_encode,ourmethod_pred= model(x)
    
        our_encode=our_encode.detach().cpu().numpy()
# #         print(HIV_A1_ourmethod_all)
#         print(ourmethod_all.shape)
#         print(ourmethod_pred)
        ourmethod_all=np.concatenate([ourmethod_all,our_encode], axis=0)
        y_all=np.concatenate([y_all,y_batch], axis=0)
        # metric on current batch
    ourmethod_all=ourmethod_all[1:,:]
    y_all=y_all[1:,:]
#     print(HIV_A1_ourmethod_all)
    
    # print(ourmethod_all)
    # print(y_all)
    ROC_AUC = metrics.roc_auc_score(y_all, ourmethod_all, multi_class='ovr')
    # MCC=matthews_corrcoef(y_all, ourmethod_all)

    print(f"Validation ROC_AUC: {ROC_AUC} ")
    df_tmp ={"Method":"VAE_finetune",
                         "ROC-AUC":ROC_AUC, "Dataset":"BACE", "Input":"Impute"}
    df.append(df_tmp)
    return df



   
def Test_one_epoch(model, test_dl):
    df = []
    model.eval()
    ourmethod_all=np.empty([1,2])
    y_all=np.empty([1,2])
    for all_data in test_dl:
        test_data_batch, y_batch=all_data

        # print(train_data_batch.x)
        optimizer.zero_grad()
        test_data_batch=test_data_batch.to(device)
        our_encode,our_decode=model(test_data_batch,)
        
        

        # y_batch_=y_batch.to(device)
#         print(x.shape)
        
        # y_encode,ourmethod_pred= model(x)
    
        our_encode=our_encode.detach().cpu().numpy()
# #         print(HIV_A1_ourmethod_all)
#         print(ourmethod_all.shape)
#         print(ourmethod_pred)
        ourmethod_all=np.concatenate([ourmethod_all,our_encode], axis=0)
        y_all=np.concatenate([y_all,y_batch], axis=0)
        # metric on current batch
    ourmethod_all=ourmethod_all[1:,:]
    y_all=y_all[1:,:]
#     print(HIV_A1_ourmethod_all)
    
    # print(ourmethod_all)
    # print(y_all)
    ROC_AUC = metrics.roc_auc_score(y_all, ourmethod_all, multi_class='ovr')
    # MCC=matthews_corrcoef(y_all, ourmethod_all)

    print(f"Test ROC_AUC: {ROC_AUC} ")
    df_tmp ={"Method":"VAE_finetune",
                         "ROC-AUC":ROC_AUC, "Dataset":"BACE", "Input":"Impute"}
    df.append(df_tmp)
    return df






print("Start our method Finetune")
for epoch in range(1, 10):
    print(f"Train Epoch {epoch}")
    train_BACE(model=model, train_loader=BACE_imputed_ourmethod_dl)
    BACE_our_impute=valid_BACE(model=model, valid_dl=BACE_valid_dl)
Test_result=Test_one_epoch(model=model, test_dl=BACE_test_dl)
print("End our method Finetune")

###### load SNN impute BACE

# print("Start SNN")
# SNN_impute = pd.read_csv("SNN_imputed_BACE.csv")
# # SNN_impute=pd.DataFrame(SNN_impute)
# # SNN_impute.shape
# # ~BACE_data["A1.0"].isna()
# # SNN_impute=SNN_impute.iloc[~BACE_data["A1.0"].isna(),:]
# # # SNN_impute.head()
# # SNN_impute=SNN_impute.iloc[]
# # sample_should_filter=(~np.isnan(y)|~BACE_data["A1.0"].isna())

# SNN_All_=SNN_impute.iloc[:,1:]
# SNN_All_.shape
# # # BACE_A2_data=BACE_A2.signature[~np.isnan(y),]
# # # BACE_A3_data=BACE_A3.signature[~np.isnan(y),]
# # # BACE_B1_data=BACE_B1.signature[~np.isnan(y),]
# # # BACE_B4_data=BACE_B4.signature[~np.isnan(y),]



# # # ##### compare with no imputation
# # # # SNN_All=np.concatenate([BACE_A1, BACE_A2_data, BACE_A3_data,BACE_B1_data ,BACE_B4_data], axis=1)
# X_SNN=SNN_All_
# SNN_All_=SNN_impute[sample_should_filter]
# y_SNN=y_nonan
# import numpy as np
# from scipy.stats import randint
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.datasets import make_moons, make_circles, make_classification
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn import metrics 
# from xgboost import XGBClassifier

# names = [
#     "Linear SVM",
#     "Random Forest",
#     "Neural Net",
#     "AdaBoost",
#     "XGBoost"
# ]

# classifiers = [
#     SVC(kernel="linear", C=0.025),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'),
# ]
# # print(X_SNN.shape)
# # print(y_SNN.shape)

# from sklearn.model_selection import train_test_split
# # i.e. 70 % training dataset and 30 % test datasets
# X_train, X_test, y_train, y_test = train_test_split(X_SNN, y_SNN, test_size = 0.30, random_state=42,stratify=y_SNN)
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=0)
# resample_noimpute_train, resample_noimpute_train_y = rus.fit_resample(X_train, y_train)

# #### resample test set
# rus_test = RandomUnderSampler(random_state=0)
# resample_noimpute_test, resample_noimpute_test_y = rus_test.fit_resample(X_test, y_test)


# SNN_impute = []
# for name, clf in zip(names, classifiers):
#         #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

#         clf = make_pipeline(StandardScaler(), clf)
#         clf.fit(resample_noimpute_train, resample_noimpute_train_y)
#         # performing predictions on the test dataset
#         y_pred = clf.predict(resample_noimpute_test)
#         ROC_AUC_natbio_128 = metrics.roc_auc_score(resample_noimpute_test_y, y_pred, multi_class='ovr')
#         # metrics are used to find accuracy or error
 
#         print()
  
#         # using metrics module for accuracy calculation
#         print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(resample_noimpute_test_y, y_pred))
#         print("ROC-AUC OF THE MODEL: ",metrics.roc_auc_score(resample_noimpute_test_y, y_pred, multi_class='ovr'))
#         no_impute_tmp ={"Method":name,
#                          "ROC-AUC":ROC_AUC_natbio_128, "Dataset":"BACE", "Input":"SNN Impute"}
#         SNN_impute.append(no_impute_tmp)