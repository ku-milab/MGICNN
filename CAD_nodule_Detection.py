import def_Patch_Extract_
import glob
import os
import numpy as np


CT_DATA_Path  = "data"
DAT_DATA_Path = "pre_process_data"
k_splits      = 5

if not os.path.exists(DAT_DATA_Path):
    os.makedirs(DAT_DATA_Path)

CT_scans = sorted(glob.glob(CT_DATA_Path + '/*.mhd'))
training, test = def_Patch_Extract_.k_fold_cross_validation(CT_scans, k_splits)

for m in range(2,3):

    # def_Patch_Extract_.slide_extraction(DAT_DATA_Path, CT_scans)
    if m == 0:
        inforPath = DAT_DATA_Path + "/" + "btm_inform.npy"
    elif m == 1:
        inforPath = DAT_DATA_Path + "/" + "mid_inform.npy"
    elif m == 2:
        inforPath = DAT_DATA_Path + "/" + "top_inform.npy"


    if not os.path.exists(inforPath):

        # lbl = def_Patch_Extract_.nodule_label_extraction(DAT_DATA_Path, CT_scans)
        # btm, mid, top = def_Patch_Extract_.total_patch_extraction(DAT_DATA_Path, CT_scans)
        # np.save(DAT_DATA_Path + "/" + "origin_inform.npy", btm, mid, top)
        btm, mid, top = def_Patch_Extract_.nodule_patch_extraction(DAT_DATA_Path, CT_scans)
        np.save(DAT_DATA_Path + "/" + "btm_inform.npy", btm)
        np.save(DAT_DATA_Path + "/" + "mid_inform.npy", mid)
        np.save(DAT_DATA_Path + "/" + "top_inform.npy",top)

    else:
        if m == 0:
            dataSize = np.load(DAT_DATA_Path + "/" + "btm_inform.npy")
        elif m == 1:
            dataSize = np.load(DAT_DATA_Path + "/" + "mid_inform.npy")
        elif m == 2:
            dataSize = np.load(DAT_DATA_Path + "/" + "top_inform.npy")

        originSize = np.load(DAT_DATA_Path + "/" + "origin_inform.npy")


    def_Patch_Extract_.run_train_3D(CT_scans, training, test, dataSize, originSize, model_num=m)

