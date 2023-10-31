#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
# import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import rdkit as rd
from rdkit import Chem


import h5py 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
import random 
from torch_cluster import knn_graph 

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
#A1
A1_sig2 = h5py.File("signature2/A1_sign2.h5", "r")
A1_feature = A1_sig2['V']
A1_keys = A1_sig2['keys']
#print(A1_keys[1:10])
#print(A1_feature.shape)
A1_feature_np=np.array(A1_feature)
A1=pd.DataFrame(data=A1_feature_np, index=A1_keys)
A1.columns = ['A1.'+str(col) for col in A1.columns]

#B1
B1_sig2 = h5py.File("signature2/B1_sign2.h5", "r")
B1_feature = B1_sig2['V']
B1_keys = B1_sig2['keys']
B1_feature_np=np.array(B1_feature)
B1=pd.DataFrame(data=B1_feature_np, index=B1_keys)
B1.columns = ['B1.'+str(col) for col in B1.columns]


# In[3]:


A2_sig2 = h5py.File("signature2/A2_sign2.h5", "r")
A2_feature = A2_sig2['V']
A2_keys = A2_sig2['keys']

A2_feature_np=np.array(A2_feature)
A2=pd.DataFrame(data=A2_feature_np
, index=A2_keys)
A2.columns = ['A2.'+str(col) for col in A2.columns]
#A2
A3_sig2 = h5py.File("signature2/A3_sign2.h5", "r")
A3_feature = A3_sig2['V']
A3_keys = A3_sig2['keys']

A3_feature_np=np.array(A3_feature)
A3=pd.DataFrame(data=A3_feature_np, index=A3_keys)
A3.columns = ['A3.'+str(col) for col in A3.columns]
#A4
A4_sig2 = h5py.File("signature2/A4_sign2.h5", "r")
A4_feature = A4_sig2['V']
A4_keys = A4_sig2['keys']

A4_feature_np=np.array(A4_feature)
A4=pd.DataFrame(data=A4_feature_np, index=A4_keys)
A4.columns = ['A4.'+str(col) for col in A4.columns]
#A5
A5_sig2 = h5py.File("signature2/A5_sign2.h5", "r")
A5_feature = A5_sig2['V']
A5_keys = A5_sig2['keys']

A5_feature_np=np.array(A5_feature)
A5=pd.DataFrame(data=A5_feature_np, index=A5_keys)
A5.columns = ['A5.'+str(col) for col in A5.columns]


# In[4]:


#B1
B1_sig2 = h5py.File("signature2/B1_sign2.h5", "r")
B1_feature = B1_sig2['V']
B1_keys = B1_sig2['keys']
B1_feature_np=np.array(B1_feature)
B1=pd.DataFrame(data=B1_feature_np, index=B1_keys)
B1.columns = ['B1.'+str(col) for col in B1.columns]
#B2
B2_sig2 = h5py.File("signature2/B2_sign2.h5", "r")
B2_feature = B2_sig2['V']
B2_keys = B2_sig2['keys']
B2_feature_np=np.array(B2_feature)
B2=pd.DataFrame(data=B2_feature_np, index=B2_keys)
B2.columns = ['B1.'+str(col) for col in B2.columns]

#B3
B3_sig2 = h5py.File("signature2/B3_sign2.h5", "r")
B3_feature = B3_sig2['V']
B3_keys = B3_sig2['keys']
B3_feature_np=np.array(B3_feature)
B3=pd.DataFrame(data=B3_feature_np, index=B3_keys)
B3.columns = ['B3.'+str(col) for col in B3.columns]


#B4
B4_sig2 = h5py.File("signature2/B4_sign2.h5", "r")
B4_feature = B4_sig2['V']
B4_keys = B4_sig2['keys']

B4_feature_np=np.array(B4_feature)
B4_sig2_key=pd.DataFrame(B4_sig2['keys'], columns=["key"])
B4=pd.DataFrame(data=B4_feature_np, index=B4_keys)
B4.columns = ['B4.'+str(col) for col in B4.columns]
#B5
B5_sig2 = h5py.File("signature2/B5_sign2.h5", "r")
B5_feature = B5_sig2['V']
B5_keys = B5_sig2['keys']

B5_feature_np=np.array(B5_feature)
B5=pd.DataFrame(data=B5_feature_np, index=B5_keys)
B5.columns = ['B5.'+str(col) for col in B5.columns]


# In[5]:


#C1
C1_sig2 = h5py.File("signature2/C1_sign2.h5", "r")
C1_feature = C1_sig2['V']
C1_keys = C1_sig2['keys']
C1_feature_np=np.array(C1_feature)
C1=pd.DataFrame(data=C1_feature_np, index=C1_keys)
C1.columns = ['C1.'+str(col) for col in C1.columns]
#C2
C2_sig2 = h5py.File("signature2/C2_sign2.h5", "r")
C2_feature = C2_sig2['V']
C2_keys = C2_sig2['keys']
C2_feature_np=np.array(C2_feature)
C2=pd.DataFrame(data=C2_feature_np, index=C2_keys)
C2.columns = ['C1.'+str(col) for col in C2.columns]

#C3
C3_sig2 = h5py.File("signature2/C3_sign2.h5", "r")
C3_feature = C3_sig2['V']
C3_keys = C3_sig2['keys']
C3_feature_np=np.array(C3_feature)
C3=pd.DataFrame(data=C3_feature_np, index=C3_keys)
C3.columns = ['C3.'+str(col) for col in C3.columns]


#C4
C4_sig2 = h5py.File("signature2/C4_sign2.h5", "r")
C4_feature = C4_sig2['V']
C4_keys = C4_sig2['keys']

C4_feature_np=np.array(C4_feature)
C4_sig2_key=pd.DataFrame(C4_sig2['keys'], columns=["key"])
C4=pd.DataFrame(data=C4_feature_np, index=C4_keys)
C4.columns = ['C4.'+str(col) for col in C4.columns]
#C5
C5_sig2 = h5py.File("signature2/C5_sign2.h5", "r")
C5_feature = C5_sig2['V']
C5_keys = C5_sig2['keys']

C5_feature_np=np.array(C5_feature)
C5=pd.DataFrame(data=C5_feature_np, index=C5_keys)
C5.columns = ['C5.'+str(col) for col in C5.columns]


# In[6]:


#D1
D1_sig2 = h5py.File("signature2/D1_sign2.h5", "r")
D1_feature = D1_sig2['V']
D1_keys = D1_sig2['keys']
D1_feature_np=np.array(D1_feature)
D1=pd.DataFrame(data=D1_feature_np, index=D1_keys)
D1.columns = ['D1.'+str(Dol) for Dol in D1.columns]
#D2
D2_sig2 = h5py.File("signature2/D2_sign2.h5", "r")
D2_feature = D2_sig2['V']
D2_keys = D2_sig2['keys']
D2_feature_np=np.array(D2_feature)
D2=pd.DataFrame(data=D2_feature_np, index=D2_keys)
D2.columns = ['D1.'+str(Dol) for Dol in D2.columns]

#D3
D3_sig2 = h5py.File("signature2/D3_sign2.h5", "r")
D3_feature = D3_sig2['V']
D3_keys = D3_sig2['keys']
D3_feature_np=np.array(D3_feature)
D3=pd.DataFrame(data=D3_feature_np, index=D3_keys)
D3.columns = ['D3.'+str(Dol) for Dol in D3.columns]


#D4
D4_sig2 = h5py.File("signature2/D4_sign2.h5", "r")
D4_feature = D4_sig2['V']
D4_keys = D4_sig2['keys']

D4_feature_np=np.array(D4_feature)
D4_sig2_key=pd.DataFrame(D4_sig2['keys'], columns=["key"])
D4=pd.DataFrame(data=D4_feature_np, index=D4_keys)
D4.columns = ['D4.'+str(Dol) for Dol in D4.columns]
#D5
D5_sig2 = h5py.File("signature2/D5_sign2.h5", "r")
D5_feature = D5_sig2['V']
D5_keys = D5_sig2['keys']

D5_feature_np=np.array(D5_feature)
D5=pd.DataFrame(data=D5_feature_np, index=D5_keys)
D5.columns = ['D5.'+str(Dol) for Dol in D5.columns]


# In[7]:


#E1
E1_sig2 = h5py.File("signature2/E1_sign2.h5", "r")
E1_feature = E1_sig2['V']
E1_keys = E1_sig2['keys']
E1_feature_np=np.array(E1_feature)
E1=pd.DataFrame(data=E1_feature_np, index=E1_keys)
E1.columns = ['E1.'+str(col) for col in E1.columns]
#E2
E2_sig2 = h5py.File("signature2/E2_sign2.h5", "r")
E2_feature = E2_sig2['V']
E2_keys = E2_sig2['keys']
E2_feature_np=np.array(E2_feature)
E2=pd.DataFrame(data=E2_feature_np, index=E2_keys)
E2.columns = ['E1.'+str(col) for col in E2.columns]

#E3
E3_sig2 = h5py.File("signature2/E3_sign2.h5", "r")
E3_feature = E3_sig2['V']
E3_keys = E3_sig2['keys']
E3_feature_np=np.array(E3_feature)
E3=pd.DataFrame(data=E3_feature_np, index=E3_keys)
E3.columns = ['E3.'+str(col) for col in E3.columns]


#E4
E4_sig2 = h5py.File("signature2/E4_sign2.h5", "r")
E4_feature = E4_sig2['V']
E4_keys = E4_sig2['keys']

E4_feature_np=np.array(E4_feature)
E4_sig2_key=pd.DataFrame(E4_sig2['keys'], columns=["key"])
E4=pd.DataFrame(data=E4_feature_np, index=E4_keys)
E4.columns = ['E4.'+str(col) for col in E4.columns]
#E5
E5_sig2 = h5py.File("signature2/E5_sign2.h5", "r")
E5_feature = E5_sig2['V']
E5_keys = E5_sig2['keys']

E5_feature_np=np.array(E5_feature)
E5=pd.DataFrame(data=E5_feature_np, index=E5_keys)
E5.columns = ['E5.'+str(col) for col in E5.columns]


# In[8]:


All_layer=A1
All_layer_name=[A2,A3,A4,A5,B1,B2,B3,B4,B5,C1,C2,C3,C4,C5,
               D1,D2,D3,D4,D5,E1,E2,E3,E4,E5]


for layer in All_layer_name:
    All_layer=pd.merge(All_layer, layer, left_index=True, right_index=True,how='left')
All_layer
# All_layer.to_csv('All_1million_small_molecule_25layers.zip')


# In[9]:


All_layer_description=All_layer.iloc[:,[1,128,128*2,128*3,128*4,128*5,128*6,128*7,128*8,128*9,128*10,128*11,
                                       128*12,
                                       128*13,128*14,128*15,128*16,128*17,128*18,128*19,128*20,128*21,128*22,128*23,128*24]]
All_layer_description.columns = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
                                'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
                                'E1','E2','E3','E4','E5']
All_layer_missing_plot=All_layer_description.notna()
# missing value visualization
import missingno as msno


# In[10]:


import seaborn as sns
sns.set_theme(style="dark")
missing_plot = sns.heatmap(
    data=All_layer_missing_plot, xticklabels=True, yticklabels='', cmap=sns.diverging_palette(220, 20, as_cmap=True))
#missing_plot.set_xticklabels(missing_plot.get_xticklabels(), rotation=45, horizontalalignment='right')


# In[11]:


missing_df=pd.DataFrame(All_layer_description.isnull().mean())
missing_df.reset_index(inplace=True)
missing_df.columns=["Layer","Missing_percentage"]
missing_df


# In[12]:


sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=missing_df, kind="bar",
    x='Layer', y="Missing_percentage",
    errorbar="sd", palette="dark", alpha=.6, height=14
)
g.despine(left=True)
g.set_axis_labels("Layer", "Missing_percentage")
g.set_xticklabels(rotation=30)
for ax in g.axes.ravel():
    
    # add annotations
    for c in ax.containers:
        labels = [f'{(v.get_height()):.3f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge')
    ax.margins(y=0.2)


#All_layer_description.isnull().mean().plot.bar(figsize=(12,6))
#plt.ylabel('Percentage of missing values')
#plt.xlabel('Layers')
#plt.title('Quantifying missing data')
#for index, row in groupedvalues.iterrows():
#    plt.text(row.name, row.tip, round(row.total_bill, 2), color='black', ha="center")


# In[13]:


# I=pd.read_csv('faiss_index.csv')

# edge = torch.tensor([I["0"], I["1"]])


# In[14]:


# ###### make sure every layer have 20% known missing value for validation
from sklearn.model_selection import train_test_split

#all_val_list=list()
train_data,validation_data = train_test_split(All_layer, test_size=0.2, random_state=1)

# from torch_geometric.nn import knn_graph


####@@@@ THis part only run once

# # Convert the data to PyTorch Geometric Data objects
# train_data_forknn = torch.tensor(train_data.fillna(0.0).values[:,0:128], dtype=torch.float)
# val_data_forknn = torch.tensor(validation_data.fillna(0.0).values[:,0:128], dtype=torch.float)

# from sklearn.neighbors import NearestNeighbors
# knn = NearestNeighbors(n_neighbors=6)

# # Fit the NearestNeighbors model with the training data
# knn.fit(train_data_forknn)
# print("Start KNN")
# distances_train, indices_train = knn.kneighbors(train_data_forknn)
# # Convert the array to a DataFrame
# indices_train_df = pd.DataFrame(indices_train)

# # Save the DataFrame to a CSV file
# indices_train_df.to_csv('train_knn.csv', index=False, header=False)

train_5nn=pd.read_csv('train_knn.csv', header=None)
# # train_5nn=train_5nn.to_numpy()
# print(train_5nn)



# # # Find the k-nearest neighbors for each validation sample
# # distances_valid, indices_valid = knn.kneighbors(val_data_forknn)
# # # print(indices_valid)
# # # Convert the array to a DataFrame
# # indices_valid_df = pd.DataFrame(indices_valid)

# # # Save the DataFrame to a CSV file
# # indices_valid_df.to_csv('valid_knn.csv', index=False, header=False)

valid_5nn=pd.read_csv('valid_knn.csv', header=None)
# # valid_5nn=valid_5nn.to_numpy()
# # print(valid_5nn)

# print("End KNN")


# In[15]:


import itertools

n = 25
lst = [i for i in range(0, n)]
combs_with_replacement = list(itertools.permutations(lst, 2))


# In[16]:


edge_index=torch.tensor(combs_with_replacement)


# In[19]:


import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import torch.nn 


        



# In[ ]:





# In[22]:


class RandomMaskingGenerator:

    def __init__(self, input_layer, A1_mask_ratio,A2_mask_ratio,A3_mask_ratio,A4_mask_ratio,A5_mask_ratio,
                 B1_mask_ratio,B2_mask_ratio,B3_mask_ratio,B4_mask_ratio,B5_mask_ratio,
                C1_mask_ratio,C2_mask_ratio,C3_mask_ratio,C4_mask_ratio,C5_mask_ratio,
                D1_mask_ratio,D2_mask_ratio,D3_mask_ratio,D4_mask_ratio,D5_mask_ratio,
                E1_mask_ratio,E2_mask_ratio,E3_mask_ratio,E4_mask_ratio,E5_mask_ratio):
        
        if not isinstance(input_layer, tuple):
            input_size = (input_layer,) * 128

        self.height= input_size

        self.num_patches = input_layer  # patch的总数即25
        self.num_mask_A1 = int(A1_mask_ratio * self.num_patches) 
        self.num_mask_A2 = int(A2_mask_ratio * self.num_patches)
        self.num_mask_A3 = int(A3_mask_ratio * self.num_patches)# 25 * 0.75
        self.num_mask_A4 = int(A4_mask_ratio * self.num_patches)  # 25 * 0.75
        self.num_mask_A5 = int(A5_mask_ratio * self.num_patches) 
        self.num_mask_B1 = int(B1_mask_ratio * self.num_patches)
        self.num_mask_B2 = int(B2_mask_ratio * self.num_patches)# 25 * 0.75
        self.num_mask_B3 = int(B3_mask_ratio * self.num_patches)  # 25 * 0.75
        self.num_mask_B4 = int(B4_mask_ratio * self.num_patches) 
        self.num_mask_B5 = int(B5_mask_ratio * self.num_patches)
        self.num_mask_C1 = int(C1_mask_ratio * self.num_patches)# 25 * 0.75
        self.num_mask_C2 = int(C2_mask_ratio * self.num_patches)  # 25 * 0.75
        self.num_mask_C3 = int(C3_mask_ratio * self.num_patches)
        self.num_mask_C4 = int(C4_mask_ratio * self.num_patches)# 25 * 0.75
        self.num_mask_C5 = int(C5_mask_ratio * self.num_patches)  # 25 * 0.75
        self.num_mask_D1 = int(D1_mask_ratio * self.num_patches) 
        self.num_mask_D2 = int(D2_mask_ratio * self.num_patches)
        self.num_mask_D3 = int(D3_mask_ratio * self.num_patches)
        self.num_mask_D4 = int(D4_mask_ratio * self.num_patches)# 25 * 0.75
        self.num_mask_D5 = int(D5_mask_ratio * self.num_patches)  # 25 * 0.75
        self.num_mask_E1 = int(E1_mask_ratio * self.num_patches) 
        self.num_mask_E2 = int(E2_mask_ratio * self.num_patches)
        self.num_mask_E3 = int(E3_mask_ratio * self.num_patches)  # 25 * 0.75
        self.num_mask_E4 = int(E4_mask_ratio * self.num_patches) 
        self.num_mask_E5 = int(E5_mask_ratio * self.num_patches)
        self.layers_list=[self.num_mask_A2,self.num_mask_A3,self.num_mask_A4,self.num_mask_A5,
                          self.num_mask_B1,self.num_mask_B2,self.num_mask_B3,self.num_mask_B4,self.num_mask_B5,
                         self.num_mask_C1,self.num_mask_C2,self.num_mask_C3,self.num_mask_C4,self.num_mask_C5,
                         self.num_mask_D1,self.num_mask_D2,self.num_mask_D3,self.num_mask_D4,self.num_mask_D5,
                         self.num_mask_E1,self.num_mask_E2,self.num_mask_E3,self.num_mask_E4,self.num_mask_E5]
       
        

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask_A1
        )
        return repr_str
    def __call__(self):
        
        mask_base = np.hstack([  # 水平方向叠起来
            np.ones(self.num_patches - self.num_mask_A1),  # 25%为0
            
            np.zeros(self.num_mask_A1),  # mask的部分设为1
        ])
        #print(mask_base)
        np.random.shuffle(mask_base)
        
        for layer in self.layers_list:
            mask_tmp = np.hstack([  # 水平方向叠起来
            np.ones(self.num_patches - layer),  # 25%为0
            
            np.zeros(layer),  # mask的部分设为1
             ])
           # print(mask_tmp)
            np.random.shuffle(mask_tmp)
        
            mask_base=np.vstack([mask_base, mask_tmp])
            
        
      
        
        #mask = np.array([mask_A1,mask_A2,mask_A3, mask_B4])
        
        return mask_base# [196]
    


# In[23]:


### base on the real missing distribution, generate a validation missing dataset
import itertools
import pandas as pd
import numpy as np
import random

def make_missing(X, frac=0.1):
    #n = int(frac * X.shape[0] * X.shape[1])
    X_list=[]

    # rows = list(range(X.shape[0]))
    # cols = list(range(X.shape[1]))

    # coordinates = list(itertools.product(*[rows, cols]))
    mask = RandomMaskingGenerator(10, missing_df["Missing_percentage"][0],missing_df["Missing_percentage"][1], missing_df["Missing_percentage"][2],
                             missing_df["Missing_percentage"][3],missing_df["Missing_percentage"][4],missing_df["Missing_percentage"][5], missing_df["Missing_percentage"][6],
                             missing_df["Missing_percentage"][7],missing_df["Missing_percentage"][8],missing_df["Missing_percentage"][9], missing_df["Missing_percentage"][10],
                             missing_df["Missing_percentage"][11],missing_df["Missing_percentage"][12],missing_df["Missing_percentage"][13], missing_df["Missing_percentage"][14],
                             missing_df["Missing_percentage"][15],missing_df["Missing_percentage"][16],missing_df["Missing_percentage"][17], missing_df["Missing_percentage"][18],
                             missing_df["Missing_percentage"][19],missing_df["Missing_percentage"][20],missing_df["Missing_percentage"][21], missing_df["Missing_percentage"][22],
                             missing_df["Missing_percentage"][23],missing_df["Missing_percentage"][24])
    mask_weuse=mask()
    mask_X=np.repeat(mask_weuse[:,5].T, 128, axis=0)
    M_ = np.copy(X)
    M_ = np.multiply(X,mask_X)

    # print(M_.shape)
    # M_=M_.clone().detach().requires_grad_(True)



    return M_, mask_X



# In[24]:


# triplet_sampler_combination_all_2=pd.read_csv('triplet_sampler_combination_all_v2.csv')
# triplet_sampler_combination_all_2=triplet_sampler_combination_all_2.to_numpy()
# triplet_sampler_combination_all_2
# triplet_sampler_combination_all_2=triplet_sampler_combination_all_2[0:100,]


# In[119]:


import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
import random 
# random.seed(10)



# class SampleDataset_triplet_sampler(Dataset):
#     def __init__(self, X,  triplet_sampler_combination_all=triplet_sampler_combination_all_2):
#         # self.__device = device
#         # self.__clazz = clazz
#         self.__X = X
#         self.combination=triplet_sampler_combination_all
        

#     def __len__(self):
# #         self.__X.shape[0]
#         return self.combination.shape[0]
#         # return 1000

#     def __getitem__(self, idx):

#         get_combination=self.combination[idx]
#         # masked_anchor=make_missing(self.__X.iloc[get_combination[0],:])

#         anchor=torch.tensor(self.__X.iloc[get_combination[0],:])
#         anchor=torch.reshape(anchor.t(), (25, 128))
#         anchor = Data(x=anchor, edge_index=edge_index.t().contiguous())



#         pos_anchor_idx=get_combination[1]
#         pos_anchor=torch.tensor(self.__X.iloc[get_combination[1],:])
#         pos_anchor=torch.reshape(pos_anchor.t(), (25, 128))
#         pos_anchor = Data(x=pos_anchor, edge_index=edge_index.t().contiguous())

#         neg_anchor_idx=get_combination[2]
#         neg_anchor=torch.tensor(self.__X.iloc[neg_anchor_idx,:])
#         neg_anchor=torch.reshape(neg_anchor.t(), (25, 128))
#         neg_anchor = Data(x=neg_anchor, edge_index=edge_index.t().contiguous())




#         return anchor,pos_anchor,neg_anchor



# In[113]:


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import *
class SampleDataset_sampler(Dataset):
    def __init__(self, X,neighbor_list=train_5nn):
        # self.__device = device
        # self.__clazz = clazz
        self.__X = X
        
        self._neighbor = neighbor_list
        # self.combination=triplet_sampler_combination_all
        

    def __len__(self):
        # self.__X.shape[0]
        return self.__X.shape[0]
        # return 10000

    def __getitem__(self, index):

        # get_combination=self.combination[idx]
        # masked_anchor=make_missing(self.__X.iloc[get_combination[0],:])
        # anchor_sample=self.__X.iloc[idx]
        get_neighbors=self._neighbor.iloc[index,1:]
        # anchor_sample_concat= [anchor_sample, self.__X.iloc[get_neighbors]]

        # anchor_sample_concat = pd.concat(anchor_sample_concat)

        anchor=torch.tensor(self.__X.iloc[index].values, dtype=torch.float32)
        # numbers_all=get_neighbors[1]
        # print(get_neighbors[1])
        numbers_train = list(range(0,5))
        # print(get_neighbors[random.choice(numbers_train)])
        # numbers.remove(idx)
        # print(random.choice(numbers))
        neg_sample=torch.tensor(self.__X.iloc[random.choice(numbers_train)].values, dtype=torch.float32)
        # print(neg_sample.shape)
        # anchor=torch.reshape(anchor.t(), (25, 128))

        # nn1=torch.tensor(self.__X.iloc[get_neighbors[1],:])
        # nn1=torch.reshape(nn1.t(), (25, 128))



        # anchor = Data(x=anchor, edge_index=edge_index.t().contiguous())





        return anchor, neg_sample






class SampleDataset_valid_sampler(Dataset):
    def __init__(self, X,neighbor_list=valid_5nn):
        # self.__device = device
        # self.__clazz = clazz
        self.__X = X
        self._neighbor = neighbor_list
        # self.train_ref=train_set
        # self.combination=triplet_sampler_combination_all
        
    def __len__(self):
#         self.__X.shape[0]
        return self.__X.shape[0]
        # return 10000

    def __getitem__(self, idx):

 
        

        anchor=torch.tensor(self.__X.iloc[idx].values, dtype=torch.float32)
        get_neighbors=self._neighbor.iloc[idx,:]
        # print(get_neighbors)
        # numbers = list(range(0, self.__X.shape[0]))
        # numbers.remove(idx)
        # neg_sample=torch.tensor(self.__X.iloc[random.choice(numbers)].values, dtype=torch.float32)
        # anchor=torch.reshape(anchor.t(), (25, 128))
        numbers_val = list(range(0,5))
        # numbers.remove(idx)
        # print(random.choice(numbers))
        neg_sample=torch.tensor(self.__X.iloc[random.choice(numbers_val)].values, dtype=torch.float32)




        # anchor = Data(x=anchor, edge_index=edge_index.t().contiguous())





        return anchor, neg_sample





# In[27]:


# gnn_train_dataset=SampleDataset(masked_train)
# gnn_train_dataloader = DataLoader(dataset = gnn_train_dataset, batch_size = 1024,drop_last=True)


# In[114]:


# import torch
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv ,SAGEConv
# from torch_geometric.utils import train_test_split_edges
# import torch.nn as nn
# from torch_geometric.nn import global_max_pool

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
    

# In[120]:


# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

train_dataset=SampleDataset_sampler(X=train_data.fillna(0.0))
# print(train_dataset[0])
number_of_workers=26
print(f"number of worker:  {number_of_workers}")
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 1024,drop_last=True,shuffle=True, num_workers=number_of_workers)


#### validation s
valid_dataset=SampleDataset_valid_sampler(X=validation_data.fillna(0.0))
# number_of_workers=26
# print(f"number of worker:  {number_of_workers}")
valid_dataloader = DataLoader(dataset = valid_dataset, batch_size = 1024,drop_last=True,shuffle=True, num_workers=number_of_workers)


# In[30]:


# first,_,_=train_dataset[0]
# set_zero=first.x
# set_zero[1:25]=0
# set_zero


# In[116]:


import torch.nn as nn
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
#         self.weights = torch.ones([3200, 1], dtype=torch.float32, device=device) 
          


 

    def forward(self, anchor_sample_frag_A,anchor_sample_frag_B,non_anchor_frag, decoded_sample, input_sample):
        
        triplet_loss =  nn.TripletMarginLoss(margin=1.0, p=2)
        
        # #print(tripet_loss)
        loss_triplet= triplet_loss(anchor_sample_frag_A,anchor_sample_frag_B, non_anchor_frag)
        loss_mse = nn.MSELoss()
        # input_sample
        mse_loss=loss_mse(decoded_sample,input_sample)
        #print(loss_triplet)
        # loss_mse = nn.MSELoss()
        # mse_loss=loss_mse(anchor_sample_frag_A,anchor_sample_frag_B)
        # loss_mse_2 = nn.MSELoss()
#         #print(loss_mse)
        
#         #target = torch.LongTensor(target)
#         #criterion = nn.CrossEntropyLoss()
#         #weight= torch.randn(3200, 1)

#         #weighted_mse_loss_value=weighted_mse_loss(output, target, weight)
# #         mse_loss=mse_loss*self.weights
#         mse_loss_2d_fingerprint=loss_mse_2(anchor, only_2d_fingerprint_encode)**10
    
       # print(mse_loss)
        return 0.5*loss_triplet+0.1*mse_loss


# In[117]:


model=Encoder(3200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
# loss_function=CustomLoss()


# In[121]:



# # Apply the mask to the node features by element-wise multiplication
# input.x = input.x * mask.unsqueeze(1)

# print(input.x.shape)


def train_one_epoch(model, optimizer, mask_A, mask_B):
    loss_function=CustomLoss()
    model.train()


    
    losses = []
    for (k, train_data_batch) in enumerate(train_dataloader):
        # print(train_data_batch)
        # train_data_batch.x, masking_data=make_missing(train_data_batch.x)
        # anchor, pos, neg=train_data_batch
        anchor, neg_sample=train_data_batch
        #### find the non-zeros part of the anchor
        # non_zero_anchor=torch.nonzero(anchor.x, as_tuple=True)
        


        #### mask nodes
        mask_anchor=anchor
        
        # print(anchor.shape)
        # non_zero_anchor=torch.nonzero(anchor,as_tuple=True)
        # print(non_zero_anchor.shape)
        # print(mask_anchor.shape)
        # print(mask_A.shape)
        mask_anchor_fragA = mask_anchor * mask_A
        mask_anchor_fragB = mask_anchor * mask_B


        mask_neg_sample=neg_sample
        mask_neg_sample= neg_sample * mask_A
        # print(mask_neg_sample)
        # mask_A= mask_A.to(device)
       

        mask_anchor_fragA=mask_anchor_fragA.to(device)
        mask_anchor_fragB=mask_anchor_fragB.to(device)
        mask_neg_sample=mask_neg_sample.to(device)
        anchor=anchor.to(device)
        
        # pos=pos.to(device)
        # neg=neg.to(device)
        # print(anchor)
   
        optimizer.zero_grad()

        pred_encoded_A, decode_A=model(mask_anchor_fragA)
        pred_encoded_B, decode_B=model(mask_anchor_fragB)
        pred_encoded_neg, decode_neg=model(mask_neg_sample)
        # print(pred_encoded_neg)
        # pos_encoded,pos_decode=model(pos)
        # neg_encoded,neg_decode=model(pos)
        # print(anchor.x)
        # print(pred_decode)

        loss=loss_function(pred_encoded_A,pred_encoded_B,pred_encoded_neg,decode_A,anchor)
            # print(loss) 
        loss.backward()

        optimizer.step()
        tmp_loss=loss.detach().cpu().numpy().item()
        print(f"Batch {k} / {len(train_dataloader)} | Train Loss = {tmp_loss}")
        losses.append(loss.detach().cpu().numpy().item())
        if k % 100 == 0 and not k == 0:
            print(f"Batch {k} / {len(train_dataloader)} | Train Loss = {tmp_loss}")
        
       
        
     
    
    # print(losses.mean())
    losses = np.array(losses)
    # print(f'epoch:{epoch + 1} '+
    #         f'loss:{losses.mean()} ')
    return losses


def validate_one_epoch(model, optimizer, mask_A, mask_B):
    model.eval()
    loss_function_valid=CustomLoss()

    


    
    losses = []
    for (k, valid_data_batch) in enumerate(valid_dataloader):
        # print(train_data_batch)
        # train_data_batch.x, masking_data=make_missing(train_data_batch.x)
        # anchor=anchor.to(device)
        # non_zero_anchor=torch.nonzero(anchor).to(device)
        anchor, neg_sample=valid_data_batch
        #### find the non-zeros part of the anchor
        # non_zero_anchor=torch.nonzero(anchor.x, as_tuple=True)


        #### mask nodes
        mask_anchor=anchor
        # print(mask_anchor.shape)
        # print(mask_A.shape)
        mask_anchor_fragA = mask_anchor * mask_A
        mask_anchor_fragB = mask_anchor * mask_B


        mask_neg_sample=neg_sample
        mask_neg_sample= neg_sample * mask_A
        # print(mask_neg_sample)
       

        mask_anchor_fragA=mask_anchor_fragA.to(device)
        mask_anchor_fragB=mask_anchor_fragB.to(device)
        mask_neg_sample =mask_neg_sample.to(device)
        anchor=anchor.to(device)
        
        # pos=pos.to(device)
        # neg=neg.to(device)
        # print(anchor)
   
        optimizer.zero_grad()

        pred_encoded_A, decode_A=model(mask_anchor_fragA)
        pred_encoded_B, decode_B=model(mask_anchor_fragB)
        pred_encoded_neg, decode_neg=model(mask_neg_sample)
        
        # print(pred_encoded_neg)
        # pos_encoded,pos_decode=model(pos)
        # neg_encoded,neg_decode=model(pos)
        # print(anchor.x)
        # print(pred_decode)

        loss=loss_function_valid(pred_encoded_A,pred_encoded_B,pred_encoded_neg,decode_A,anchor)
            # print(loss) 
        loss.backward()

        optimizer.step()
        tmp_loss=loss.detach().cpu().numpy().item()
        losses.append(loss.detach().cpu().numpy().item())
        if k % 100 == 0 and not k == 0:
            print(f"Validation Loss = {tmp_loss}")
        
        
        
     
    
    # print(losses.mean())
    losses = np.array(losses)
    # print(f'epoch:{epoch + 1} '+
    #         f'valid loss:{losses.mean()} ')
    return losses
 


# In[109]:


# from tqdm import tqdm


# # In[125]:



epochs=20
print("Start Training")
for epoch in range(epochs):
    mask = RandomMaskingGenerator(10000, missing_df["Missing_percentage"][0],missing_df["Missing_percentage"][1], missing_df["Missing_percentage"][2],
                             missing_df["Missing_percentage"][3],missing_df["Missing_percentage"][4],missing_df["Missing_percentage"][5], missing_df["Missing_percentage"][6],
                             missing_df["Missing_percentage"][7],missing_df["Missing_percentage"][8],missing_df["Missing_percentage"][9], missing_df["Missing_percentage"][10],
                             missing_df["Missing_percentage"][11],missing_df["Missing_percentage"][12],missing_df["Missing_percentage"][13], missing_df["Missing_percentage"][14],
                             missing_df["Missing_percentage"][15],missing_df["Missing_percentage"][16],missing_df["Missing_percentage"][17], missing_df["Missing_percentage"][18],
                             missing_df["Missing_percentage"][19],missing_df["Missing_percentage"][20],missing_df["Missing_percentage"][21], missing_df["Missing_percentage"][22],
                             missing_df["Missing_percentage"][23],missing_df["Missing_percentage"][24])
    mask_weuse=mask()
    # Create a mask to mask the second node
    # mask_ = torch.tensor([1,0,0,0,0,0,0,0,0,0,
    #                       0,0,0,0,0,0,0,0,0,0,
    #                       0,0,0,0,0,
    #                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    #                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    #                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    #                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    #                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=torch.float)
    num1 = random.randint(0, 10000)
    # mask_A = torch.tensor(mask_weuse[:,num1], dtype=torch.float)
    mask_A=torch.tensor(np.repeat(mask_weuse[:,num1].T, 128, axis=0), dtype=torch.float)
    num2 = random.randint(0, 10000)
    mask_B = torch.tensor(np.repeat(mask_weuse[:,num2].T, 128, axis=0), dtype=torch.float)  
    # num3 = random.randint(0, 1000)
    # mask_C = torch.tensor(np.repeat(mask_weuse[:,num1].T, 128, axis=0), dtype=torch.float)  
  
    
    losses= train_one_epoch(model, optimizer, mask_A, mask_B)
    valid_losses= validate_one_epoch(model, optimizer,mask_A,mask_B)
    print(f'epoch:{epoch + 1} '+
            f'loss:{losses.mean()} '+
            f'validation loss:{valid_losses.mean()}')
    # print(f'epoch:{epoch + 1} '+
    #         f'loss:{losses.mean()} ')
    
    torch.save({
            'epoch': epoch ,
            'model_state_dict': model
.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            }, 'encoder_model_v2.pt')

print("End Training")


# In[109]:


# from tqdm import tqdm


# # In[125]:


# epochs=20
# print("Start Training")
# for epoch in range(epochs):
    
#     losses= train_one_epoch(model, optimizer)
#     torch.save({
#             'epoch': epoch ,
#             'model_state_dict': model
# .state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': losses,
#             }, 'GNN_model.pt')

# print("End Training")


# In[ ]:


# # parameters
# out_channels = 18
# num_features = 3200
# epochs = 100

# # model
# # model = GCNEncoder_decoder(num_features)
# model = GAE(GCNEncoder_decoder(num_features))

# # move to GPU (if available)

# # model = model.to(device)
# # x = torch.tensor(data.x).to(torch.float32).to(device)
# # train_pos_edge_index = data.train_pos_edge_index.to(device)

# # inizialize the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[ ]:


# import torch.nn as nn
# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
# #         self.weights = torch.ones([3200, 1], dtype=torch.float32, device=device) 

 

#     def forward(self, output, target,anchor, pos, neg, only_2d_fingerprint_encode):
        
#         triplet_loss =  nn.TripletMarginLoss(margin=1.0, p=2)
#         #print(tripet_loss)
#         loss_triplet= triplet_loss(anchor, pos, neg)
#         #print(loss_triplet)
#         loss_mse = nn.MSELoss()
#         loss_mse_2 = nn.MSELoss()
#         #print(loss_mse)
        
#         #target = torch.LongTensor(target)
#         #criterion = nn.CrossEntropyLoss()
#         #weight= torch.randn(3200, 1)
#         mse_loss=loss_mse(output, target)
#         #weighted_mse_loss_value=weighted_mse_loss(output, target, weight)
# #         mse_loss=mse_loss*self.weights
#         mse_loss_2d_fingerprint=loss_mse_2(anchor, only_2d_fingerprint_encode)**10
    
#        # print(mse_loss)
#         return mse_loss + loss_triplet+mse_loss_2d_fingerprint


# In[ ]:


# def Mask_filler(items): 
    
#     masked_items, coordinates = make_missing(items)
#     masked_items=masked_items.fillna(0.0)
#     masked_items=masked_items.values
#     masked_items = torch.from_numpy(masked_items).to(torch.float32)
    
#     # masked_pos_anchor, coordinates_pos_anchor = make_missing(pos_anchor)
#     # masked_pos_anchor=masked_pos_anchor.fillna(0.0)
#     # masked_pos_anchor=masked_pos_anchor.values
#     # masked_pos_anchor = torch.from_numpy(masked_pos_anchor).to(torch.float32)
    
#     # masked_neg_anchor, coordinates_neg_anchor = make_missing(neg_anchor)
#     # masked_neg_anchor=masked_neg_anchor.fillna(0.0)
#     # masked_neg_anchor=masked_neg_anchor.values
#     # masked_neg_anchor = torch.from_numpy(masked_neg_anchor).to(torch.float32)
    
#     return masked_items
    


# In[ ]:


# loss_mse = nn.MSELoss()
# def train():
#     model.train()
#     optimizer.zero_grad()
#     masked_x= Mask_filler(data.x).to(device)
#     encoded, reconstructed = model.encode(masked_x, data.edge_index.to(device))
#     # loss = model.recon_loss(z, train_pos_edge_index)
#     loss=loss_mse(x, reconstructed)
#     #if args.variational:
#     #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# In[ ]:


# data.edge


# In[ ]:


# model.train()
# optimizer.zero_grad()
# masked_x= Mask_filler(data.x).to(device)
# encoded, reconstructed = model(masked_x, data.edge_index.to(device))
# # for epoch in range(100):
# #     print(epoch)
# #     train()

