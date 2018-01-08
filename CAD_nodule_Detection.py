
import glob
import os
import numpy as np

import def_Patch_Extract_
import train_model

if preprocessing_Flag == 1:
    CT_DATA_Path = "data"
    DAT_DATA_Path = "pre_process_data"
    k_splits = 5

    if not os.path.exists(DAT_DATA_Path):
        os.makedirs(DAT_DATA_Path)

    CT_scans = sorted(glob.glob(CT_DATA_Path + '/*.mhd'))
    training, test = def_Patch_Extract_.k_fold_cross_validation(CT_scans, k_splits)

    for m in range(0,3):

        # def_Patch_Extract_.slide_extraction(DAT_DATA_Path, CT_scans)
        if m == 0:
            inforPath = DAT_DATA_Path + "/" + "btm_inform.npy"
        elif m == 1:
            inforPath = DAT_DATA_Path + "/" + "mid_inform.npy"
        elif m == 2:
            inforPath = DAT_DATA_Path + "/" + "top_inform.npy"


        if os.path.exists(inforPath):

            # lbl = def_Patch_Extract_.nodule_label_extraction(DAT_DATA_Path, CT_scans)
            btm, mid, top = def_Patch_Extract_.test_patch_extraction(DAT_DATA_Path, CT_scans)
            np.save(DAT_DATA_Path + "/" + "btm_test_inform.npy", btm)
            np.save(DAT_DATA_Path + "/" + "mid_test_inform.npy", mid)
            np.save(DAT_DATA_Path + "/" + "top_test_inform.npy", top)
            btm, mid, top = def_Patch_Extract_.nodule_patch_extraction(DAT_DATA_Path, CT_scans, modeFlag=False)
            np.save(DAT_DATA_Path + "/" + "btm_inform.npy", btm)
            np.save(DAT_DATA_Path + "/" + "mid_inform.npy", mid)
            np.save(DAT_DATA_Path + "/" + "top_inform.npy",top)

        else:
            if m == 0:
                dataSize = np.load(DAT_DATA_Path + "/" + "btm_inform.npy")
                originSize = np.load(DAT_DATA_Path + "/" + "btm_test_inform.npy")
            elif m == 1:
                dataSize = np.load(DAT_DATA_Path + "/" + "mid_inform.npy")
                originSize = np.load(DAT_DATA_Path + "/" + "mid_test_inform.npy")
            elif m == 2:
                dataSize = np.load(DAT_DATA_Path + "/" + "top_inform.npy")
                originSize = np.load(DAT_DATA_Path + "/" + "top_test_inform.npy")

        def_Patch_Extract_.mk_train_dataset(CT_scans, training, test, dataSize, originSize, model_num=m)

else:

    dataset = " "
    irs_dataset = " "

    model_def = train_model.model_def()
    model_exec = train_model.model_execute(dataset=dataset, irs_dataset=irs_dataset)

    if Multi_scale_train:
        cross_entropy, softmax, layers1, layers2, data_node1, data_node2, top_node1, top_node2, bottom_node1, \
        bottom_node2, label_node = model_def.bck_Complex_RM_CNN(train=True, is_leaky=False)

        model_exec.train_bck_Complex_RM_CNN(cand_num=cand_Flag, cross_entropy=cross_entropy, softmax=softmax,
                                            data_node1=data_node1,data_node2=data_node2,
                                            top1=top_node1, top2=top_node2,
                                            bottom1=bottom_node1, bottom2=bottom_node2, label_node=label_node)
    if Multi_scale_test:
        cross_entropy, softmax, layers1, layers2, data_node1, data_node2, top_node1, top_node2,\
        bottom_node1, bottom_node2, label_node = model_def.bck_Complex_RM_CNN(train=False, is_leaky=False)
        pm_path = model_exec.test_bck_Complex_RM_CNN(cand_num=cand_Flag, softmax=softmax,
                                                     data_node1=data_node1,data_node2=data_node2,
                                                     top1=top_node1, top2=top_node2,
                                                     bottom1=bottom_node1, bottom2=bottom_node2, md_num='38')

        model_num = 4 # 3: cand V1 / 4: cand V2
        model_exec.code_test(model_num, model_name=pm_path)

        if model_num == 4:
            cands = def_Patch_Extract_.readCSV(filename="CSVFile/candidates_V2.csv")
        elif model_num !=4:
            cands = def_Patch_Extract_.readCSV(filename="CSVFile/candidates.csv")

        output_dir, CSV_filename = train_model.mk_CSVwriter(cands, model_num, model_name=pm_path)
        train_model.mk_FROC_results(output_dir, CSV_filename)
        train_model.CPM_results(output_dir, CSV_filename)
