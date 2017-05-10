import def_Patch_Extract_
import glob
import os
import numpy as np


CT_DATA_Path = "data"
DAT_DATA_Path = "pre_process_data"
if not os.path.exists(DAT_DATA_Path):
    os.makedirs(DAT_DATA_Path)

CT_scans = sorted(glob.glob(CT_DATA_Path + '/*.mhd'))

# def_Patch_Extract_.slide_extraction(DAT_DATA_Path, CT_scans)

if not os.path.exists(DAT_DATA_Path + "/" + "inform.npy"):


    # lbl = def_Patch_Extract_.nodule_label_extraction(DAT_DATA_Path, CT_scans)
    # btm, mid, top = def_Patch_Extract_.total_patch_extraction(DAT_DATA_Path, CT_scans)
    # np.save(DAT_DATA_Path + "/" + "origin_inform.npy", btm, mid, top)
    btm, mid, top = def_Patch_Extract_.nodule_patch_extraction(DAT_DATA_Path, CT_scans)
    np.save(DAT_DATA_Path + "/" + "inform.npy", btm, mid, top)

else:

    dataSize = np.load(DAT_DATA_Path + "/" + "inform.npy")
    originSize = np.load(DAT_DATA_Path + "/" + "origin_inform.npy")

k_splits = 5
training, validation, test = def_Patch_Extract_.k_fold_cross_validation(CT_scans, k_splits)

# def_Patch_Extract_.run_train_3D(CT_scans, training, validation, test, dataSize)
def_Patch_Extract_.run_train_3D(CT_scans, training, validation, test, dataSize, originSize)
# def_Patc-h_Extract_.run_train(training, validation, test, dataSize)
