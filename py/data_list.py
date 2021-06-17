
# %%
import os
import re
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime

# %%
data_path = r"C://Users//Jean-Baptiste//OneDrive//ENSAE//2A//CHU//Prediction_soustype//data"
meta_data_path = r"C://Users//Jean-Baptiste//OneDrive//ENSAE//2A//CHU//Prediction_soustype//data//Recueil patients Blastes LAB annoté.xlsx"
meta = pd.read_excel(meta_data_path)
#%%
meta[meta.Type == 'LAL B1']
 # %%
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
meta.No=meta.No.apply(num_pat)
meta.set_index('No',inplace=True)

# %%
patient_no = [k for k in listdir(data_path) if not isfile(join(data_path, k))]
date_diag1= meta['Diagnostic'].apply(lambda x: x.strftime('%d')+ x.strftime('%m')+x.strftime('%Y')[-2:])
date_rc1 = meta['RC diagnostic'].apply(lambda x: x.strftime('%d')+ x.strftime('%m')+x.strftime('%Y')[-2:])
date_diag = date_diag1[patient_no].to_dict()
date_rc = date_rc1[patient_no].to_dict()

# %%
patient={}
for k in date_diag.keys():
    patient[num_pat(k)]=[num_pat(k),date_diag[k],date_rc[k], meta['Type'][k]]
# %%
fold_patients = {}
sub_fold = list(filter(os.path.isdir, [os.path.join(data_path, f) for f in os.listdir(data_path)]))
dic_subfolders = {}
for k in range(len(patient_no)):
    dic_subfolders[patient_no[k]] = sub_fold[k]

for k in patient_no:
    L = []
    for fold in list(filter(os.path.isdir, [os.path.join(dic_subfolders[k], f) for f in os.listdir(dic_subfolders[k])])):
        if date_sup(fold[-6:], patient[k][1]) and not date_sup(fold[-6:], patient[k][2]):
            L.append(fold)
    fold_patients[k] = L

# %%
im_data = []
files_patients= {}

for pat in patient_no:
    list_image_pat=[]
    name,_,type,group,_,_,_,_,_,_,_,_,_ = meta.loc[pat]
    for fold in fold_patients[pat]:
        list_images_in_fold = os.listdir(fold)
        if len(list_images_in_fold)==0:
            continue
        for image in os.listdir(fold):
            full_path = os.path.join(fold, image)
            list_image_pat.append(full_path)
    folder_meta = dict(PATIENT_NO = pat, PATIENT_NAME=name, PATIENT_TYPE=type, PATIENT_GROUP=group, IMAGES=list_image_pat, FOLDERS = fold_patients[pat])
    im_data.append(folder_meta)
    files_patients[pat] = list_image_pat

# im_data.append(dict(files_patients))
im_data = pd.DataFrame(im_data)
im_data.to_csv(r"C://Users//Jean-Baptiste//OneDrive//ENSAE//2A//CHU//Prediction_soustype//data//data_patient.csv", index = False)




# %%
# files_patients= {}
# for pat in patient_no:
    
#     for fold in fold_patients[pat]:
#         list_images_in_dir = os.listdir(fold)
#         if len(list_images_in_dir)!=0:
#             # path_image = os.path.join(fold_patients[pat], fold)
#             L = L+list_images_in_dir
        
#     files_patients[pat]=L

 # %%
im_data[im_data]
# %%
files_patients['0054']
# %%

# %%
os.listdir("C://Users//Jean-Baptiste//OneDrive//ENSAE//2A//CHU//Prediction_soustype//data//0051//18541958204-051118")