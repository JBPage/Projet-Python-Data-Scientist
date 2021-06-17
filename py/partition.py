#%%
import numpy as np
from numpy.lib.twodim_base import triu_indices_from
import pandas as pd
import os
import pickle
import re
from sklearn.utils import validation
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split

import random
#%%
# load metadata
metadata = pd.read_excel(r"C://Users//Jean-Baptiste//OneDrive//ENSAE//2A//CHU//Prediction_soustype//data//Recueil patients Blastes LAB annoté.xlsx")
data_patient = pd.read_csv(r"C://Users//Jean-Baptiste//OneDrive//ENSAE//2A//CHU//Prediction_soustype//data//data_patient.csv")
#%%
# load list of patients with folders that can be used (between diagnostic and remission) & list of images
im_path = r"C:/Users/Jean-Baptiste/OneDrive/ENSAE/2A/CHU/Prediction_soustype/data"
im_test_path = r"C:/Users/Jean-Baptiste/OneDrive/ENSAE/2A/CHU/Prediction_soustype/data_test"

def num_pat(x):
    a=str(x)
    while len(a)<4:
        a = '0'+a
    return(a)
def date_sup(date_1, date_2):
    #dit si date1 est post à date2
    if date_1[-2:]>date_2[-2:]:
        return(True)
    if  date_1[-2:]<date_2[-2:]:
        return(False)
    else:
        if date_1[2:-2]>date_2[2:-2]:
            return(True)
        if  date_1[2:-2]<date_2[2:-2]:
            return(False)
        else:
            if date_1[:2]>date_2[:2]:
                return(True)
            if  date_1[:2]<date_2[:2]:
                return(False)
metadata.No=metadata.No.apply(num_pat)
metadata.set_index('No',inplace=True)
patient_no = [k for k in metadata.index if k in os.listdir(im_path)]
data_patient['FOLDERS']  = data_patient['FOLDERS'].apply(lambda x: ast.literal_eval(x))
data_patient['IMAGES']  = data_patient['IMAGES'].apply(lambda x: ast.literal_eval(x))
data_patient['PATIENT_NO'] = data_patient['PATIENT_NO'].apply(lambda x: num_pat(x))
data_patient.set_index('PATIENT_NO', inplace= True)
#%%
images_metadata = data_patient.to_dict('index')
images_metadata = {k:v for k,v in images_metadata.items() if len(v['IMAGES'])>0}
#%%

for k,v in images_metadata.items():
    if v['PATIENT_TYPE'] == 'LAL B2':
        new_image = v['IMAGES'][:len(v['IMAGES'])//2]
        v['IMAGES'] = new_image

#%%
#dict with all images name path and type
# grouped_images_metadata = {curgroup:[v for k,v  in images_metadata.items() if v['PATIENT_TYPE']==curgroup] for curgroup in ('LAL B1','LAL B2','LAL B3')}
grouped_images_metadata = {}
for curgroup in ('LAL B1','LAL B2','LAL B3'):
    grouped_images_metadata_curgroup = []
    for k,el in images_metadata.items():
        if el['PATIENT_TYPE'] == curgroup:
            for path_image in list(el['IMAGES']):
                name_image = os.path.basename(path_image)
                grouped_images_metadata_curgroup.append([name_image,path_image, curgroup])
    grouped_images_metadata[curgroup] = grouped_images_metadata_curgroup
#%%
p=0
for k,v in images_metadata.items():
    if v['PATIENT_TYPE'] == 'LAL B2':
        p+=len(v['IMAGES'])
print(p)
#%%
part_rng = np.random.RandomState(seed=45)

# part each group
for curgroup in ('LAL B1','LAL B2','LAL B3'):
    samples_is = np.arange(len(grouped_images_metadata[curgroup]))
    train_part = part_rng.choice(a = samples_is, size = len(samples_is)//2, replace = False)
    valid_part = part_rng.choice(a = np.setdiff1d(samples_is, train_part), size = (len(samples_is)-train_part.shape[0])//2, replace = False)
    test_part = np.setdiff1d(np.setdiff1d(samples_is, train_part), valid_part)
    for i in train_part:
        grouped_images_metadata[curgroup][i].append('train')
    for i in valid_part:
        grouped_images_metadata[curgroup][i].append('valid')
    for i in test_part:
        grouped_images_metadata[curgroup][i].append('test')




#%%
p=0
for k,v in grouped_images_metadata.items():
    for el in v:
        if el[2]== 'LAL B2':
            p+=1
print(p)
#%%
#number of images per LAL type
# LAL B1 : 0
# LAL B2 : 12124
# LAL B3 : 1204


#%%
# merge back
all_metadata = [data for curgroup,elem in grouped_images_metadata.items() for data in elem ]

# debug_df = pd.DataFrame(all_metadata)
# debug_df.groupby(['GROUP','PART']).sum()
#%%

# debug_df.groupby(['NAME','PART']).sum()

# create output


meta = dict(train=[],valid=[],test=[])

for metadata in all_metadata:
    tmp_sublist = [dict(FILE=metadata[0],PATH=metadata[1],GROUP=metadata[2])]
    meta[metadata[3]].extend(tmp_sublist)
#%%

for curpart in ('train','valid','test'):
    with open(os.path.join(im_path,"part_{}.pkl".format(curpart)), 'wb') as file_pi:
        pickle.dump(meta[curpart], file_pi)
        
from PIL import Image
        
# get max size of image
max_w=0
max_h=0
for curpart in ('train','valid','test'):
    for meta_elem in tqdm(meta[curpart]):
        cur_w,cur_h=Image.open(os.path.join(im_path,meta_elem['PATH'])).size
        max_w=max(max_w,cur_w)
        max_h=max(max_h,cur_h)
print('Max dimensions are: {} x {}'.format(max_w,max_h)) # 263 x 299 # let's choose 320 (10*2^5)


# %%
#debug
for k in ('train', 'test', 'valid'):
    for j in ('LAL B1','LAL B2','LAL B3'):
        a=0
        for el in meta[k]:
            if el['GROUP']==j:
                a+=1
        print(j,k,a)
        
# LAL B1 train 0
# LAL B2 train 3027
# LAL B3 train 300
# LAL B1 test 0
# LAL B2 test 1514
# LAL B3 test 151
# LAL B1 valid 0
# LAL B2 valid 1514
# LAL B3 valid 150
