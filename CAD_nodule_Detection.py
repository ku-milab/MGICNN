
import def_Patch_Extract_
import glob
import os
import numpy as np

CT_DATA_Path = "data"
DAT_DATA_Path = "output"
if not os.path.exists(DAT_DATA_Path):
    os.makedirs(DAT_DATA_Path)

CT_scans = sorted(glob.glob(CT_DATA_Path + '/*.mhd'))

if os.path.exists(DAT_DATA_Path + "/" + "inform.npy"):
    data_size = def_Patch_Extract_.nodule_patch_extraction(DAT_DATA_Path, CT_scans)
    np.save(DAT_DATA_Path + "/" + "inform.npy", data_size)
else:
    data_size = np.load(DAT_DATA_Path + "/" + "inform.npy")

k_splits = 5
training, validation, test = def_Patch_Extract_.k_fold_cross_validation(CT_scans, k_splits)

def_Patch_Extract_.run_train(training, validation, test, data_size)