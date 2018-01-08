import csv
import os
import numpy as np

import matplotlib.pyplot as plt

from random import shuffle
import SimpleITK as sitk
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy import ndimage as nd

import train_model
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly.graph_objs import *

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage) # z, y, x axis load
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    # print(numpyOrigin, numpySpacing)
    # numpyImage = np.transpose(numpyImage, axes=(1, 2, 0))
    # numpyOrigin = [numpyOrigin[2], numpyOrigin[1], numpyOrigin[0]]
    # numpySpacing = [numpySpacing[2], numpySpacing[1], numpySpacing[0]]
    # print(numpyOrigin, numpySpacing)

    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines[1:]

def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[ npzarray > 1 ] = 1.
    npzarray[ npzarray < 0 ] = 0.
    return npzarray

def img_rotation(img, k):

    rot_img = np.rot90(img, k, axes=(1,2))

    return rot_img

def org_VoxelCoord(lists, str, origin, spacing):
    cand_list = []
    labels = []

    for list in lists:
        if list[0] in str:
            worldCoord = np.asarray([float(list[3]), float(list[2]), float(list[1])])
            voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
            cand_list.append(voxelCoord)
            labels.append(int(list[4]))

    return cand_list, labels

def chg_VoxelCoord(lists, str, origin, spacing):
    cand_list = []
    labels = []
    # if len(lists) > 2000:
    for list in lists:
        if list[0] in str:
            worldCoord = np.asarray([float(list[3]), float(list[2]), float(list[1])])
            voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
            if list[4] is '1':
                augs, aug_labels = aug_candidate(voxelCoord)
                cand_list.append(voxelCoord)
                labels.append(int(list[4]))
                for aug in augs:
                    cand_list.append(aug)
                al_vec = np.ones((int(aug_labels),1))
                for aug_lbl in al_vec:
                    labels.append(int(aug_lbl))
            else:
                cand_list.append(voxelCoord)
                labels.append(int(list[4]))
    return cand_list, labels

def VoxelCoord(lists, str, origin, spacing):
    cand_list = []

    for list in lists:
        if list[0] in str:
            worldCoord = np.asarray([float(list[1]), float(list[2]), float(list[3])])
            voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
            cand_list.append([voxelCoord[0], voxelCoord[1], voxelCoord[2]])
            print('|', worldCoord, '|->|', voxelCoord, '|')

    return cand_list

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def aug_candidate(list):
    #aug = []
    shiftCoord = [[1, 1, 1],   [1, 1, 0],   [1, 1, -1],
                  [1, 0, 1],   [1, 0, 0],   [1, 0, -1],
                  [1, -1, 1],  [1, -1, 0],  [1, -1, -1],
                  [0, 1, 1],   [0, 1, 0],   [0, 1, -1],
                  [0, 0, 1],   [0, 0, -1],  [0, -1, 1],
                  [0, -1, 0],  [0, -1, -1], [-1, 1, 1],
                  [-1, 1, 0],  [-1, 1, -1], [-1, 0, 1],
                  [-1, 0, 0],  [-1, 0, -1], [-1, -1, 1],
                  [-1, -1, 0], [1, -1, -1]]

    aug = list + shiftCoord
    aug_size = len(aug)
    return aug, aug_size

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    #for i in range(k):
    # validation = slices[0]
    test = slices[0]
    training = slices[1] + slices[2] + slices[3] + slices[4]
    # return training, validation, test
    return training, test

def nodule_label_extraction(DAT_DATA_Path, CT_scans):
    cands = readCSV(filename="CSVFile/candidates.csv")
    annot = readCSV(filename="CSVFile/annotations.csv")
    lbl = []

    for a in annot:
        for c, cand in enumerate(cands):
            if cand[-1] == '1':
                if a[0] == cand[0]:
                    if cal_diameter(a[1:-1], cand[1:-1], a[-1]):
                        lbl.append(c)
                        print(c)
                        # else:
                        #     lbl.append('0')

    return lbl

def nodule_patch_extraction(DAT_DATA_Path, CT_scans, modeFlag):
    cands = readCSV(filename="CSVFile/candidates_V2.csv")

    btm_SHApes = []
    mid_SHApes = []
    top_SHApes = []

    DAT_Path = DAT_DATA_Path
    voxelWidth = [[6, 20, 20, 'btm'], [10, 30, 30, 'mid'], [26, 40, 40, 'top']]

    for v, vw in enumerate(voxelWidth):

        if modeFlag:
            VW_std = [26, 40, 40]
        else:
            VW_std = vw
        DAT_DATA_Path = DAT_Path
        DAT_DATA_Path = DAT_DATA_Path + '/%s' % (vw[3])

        if not os.path.exists(DAT_DATA_Path):
            os.mkdir(DAT_DATA_Path)
        print(DAT_DATA_Path)
        for ct, img_dir in enumerate(CT_scans):
            npImage, npOrigin, npSpacing = load_itk_image(img_dir)
            normImage = normalizePlanes(npImage)
            voxelCands, labels = chg_VoxelCoord(cands, img_dir, npOrigin, npSpacing)

            candNum = labels.count(0) + (labels.count(1) * 4)

            pData = np.memmap(filename=DAT_DATA_Path + "/temp.dat",
                              dtype='float32', mode='w+', shape=(vw[0], vw[1], vw[2], candNum))
            plabel = np.memmap(filename=DAT_DATA_Path + "/temp.lbl",
                               dtype='uint8', mode='w+', shape=(1, 1, candNum))
            cNum = 0
            for i, cand in enumerate(voxelCands):
                Cond_Arg = [(math.floor(cand[0]) - VW_std[0] / 2), (math.floor(cand[0]) + VW_std[0] / 2),
                            (math.floor(cand[1]) - VW_std[1] / 2), (math.floor(cand[1]) + VW_std[1] / 2),
                            (math.floor(cand[2]) - VW_std[2] / 2), (math.floor(cand[2]) + VW_std[2] / 2)]

                arg_arange = [(math.floor(cand[0]) - vw[0] / 2), (math.floor(cand[0]) + vw[0] / 2),
                              (math.floor(cand[1]) - vw[1] / 2), (math.floor(cand[1]) + vw[1] / 2),
                              (math.floor(cand[2]) - vw[2] / 2), (math.floor(cand[2]) + vw[2] / 2)]

                if Compare_Array(Cond_Arg, normImage.shape):
                    patch = TP_voxel_extraction(Img=normImage, arg=arg_arange, voxelWidth=vw)

                    if patch.shape[0] != pData.shape[0] or patch.shape[1] != pData.shape[1] \
                            or patch.shape[2] != pData.shape[2] and labels[i] != 1:
                        print("voxelWidth: %d, imageNum: %d, candNum: %d, label: %d \n" % (v, ct, i, labels[i]))
                    elif patch.shape[0] != pData.shape[0] or patch.shape[1] != pData.shape[1] \
                            or patch.shape[2] != pData.shape[2] and labels[i] == 1:
                        print("voxelWidth: %d, imageNum: %d, candNum: %d, label: %d \n" % (v, ct, i, labels[i]))
                    else:
                        pData[:, :, :, cNum] = patch.copy()
                        plabel[:, :, cNum] = labels[i]
                        cNum += 1
                        if labels[i] == 1:
                            for k in range(1, 4):
                                rot_patch = img_rotation(patch, k)
                                pData[:, :, :, cNum] = rot_patch.copy()
                                plabel[:, :, cNum] = labels[i]
                                cNum += 1

            pTempData = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".dat",
                                  dtype='float32', mode='w+', shape=(vw[0], vw[1], vw[2], cNum))
            pTemplabel = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".lbl",
                                   dtype='uint8', mode='w+', shape=(1, 1, cNum))
            pTempData[:, :, :, :] = pData[:, :, :, :cNum]
            pTemplabel[:, :, :] = plabel[:, :, :cNum]

            del pData, plabel, pTempData, pTemplabel

            if v == 0:
                btm_SHApes.append([img_dir[5:-4], cNum])
            elif v == 1:
                mid_SHApes.append([img_dir[5:-4], cNum])
            elif v == 2:
                top_SHApes.append([img_dir[5:-4], cNum])


    return btm_SHApes, mid_SHApes, top_SHApes

def TP_voxel_extraction(Img, arg, voxelWidth):

    if arg[0] <= 0 and abs(0. - arg[0]) <= 4:
        scale = abs(1. - arg[0])
        arg[0] = arg[0] + scale
        arg[1] = arg[0] + voxelWidth[0]
        print(scale, 'arg0-1')

    if arg[2] <= 0 and abs(0. - arg[2]) <= 4:
        scale = abs(1. - arg[2])
        arg[2] = arg[2] + scale
        arg[3] = arg[2] + voxelWidth[1]
        print(scale, 'arg2-3')

    if arg[4] <= 0 and abs(0. - arg[4]) <= 4:
        scale = abs(1. - arg[4])
        arg[4] = arg[4] + scale
        arg[5] = arg[4] + voxelWidth[2]
        print(scale, 'arg4-5')

    if arg[1] > Img.shape[0] and abs(Img.shape[0] - arg[1]) <= 4:
        scale = abs(Img.shape[0] - arg[1])
        arg[1] = arg[1] - scale
        arg[0] = arg[1] - voxelWidth[0]
        print(scale, 'arg1-0')

    if arg[3] > Img.shape[1] and abs(Img.shape[1] - arg[3]) <= 4:
        scale = abs(Img.shape[1] - arg[3])
        arg[3] = arg[3] - scale
        arg[2] = arg[3] - voxelWidth[1]
        print(scale, 'arg3-2')

    if arg[5] > Img.shape[2] and abs(Img.shape[2] - arg[5]) <= 4:
        scale = abs(Img.shape[2] - arg[5])
        arg[5] = arg[5] - scale
        arg[4] = arg[5] - voxelWidth[2]
        print(scale, 'arg5-4')

    voxelTensor = np.array(Img[int(arg[0]):int(arg[1]),
                           int(arg[2]):int(arg[3]),
                           int(arg[4]):int(arg[5])])
    return voxelTensor

def Compare_Array(arr1, arr2):
    res = True

    for a, arr in enumerate(arr1):
        if a == 0 or a == 1:
            res = res * (arr >= 0) * (arr <= arr2[0])
        else:
            res = res * (arr >= 0) * (arr <= 511)

    return res

def total_patch_extraction(DAT_DATA_Path, CT_scans):
    cand_path = "CSVFile/candidates.csv"
    # cand_path = "CSVFile/candidates_V2.csv"
    cands = readCSV(cand_path)

    btm_SHApes = []
    mid_SHApes = []
    top_SHApes = []

    voxelWidth = [[6, 20, 20, 'btm'], [10, 30, 30, 'mid'], [26, 40, 40, 'top']]
    # VWidth_std = [6, 20, 20]

    for v, vw in enumerate(voxelWidth):
        DAT_DATA_Path = 'origin_voxel_V1'
        if not os.path.exists(DAT_DATA_Path):
            os.mkdir(DAT_DATA_Path)

        DAT_DATA_Path = DAT_DATA_Path + '/%s' % (vw[3])
        if not os.path.exists(DAT_DATA_Path):
            os.mkdir(DAT_DATA_Path)

        print(DAT_DATA_Path)
        for ct, img_dir in enumerate(CT_scans):
            npImg, npOrigin, npSpacing = load_itk_image(img_dir)
            normImage = normalizePlanes(npImg)
            print(normImage.shape)
            voxelCands, labels = org_VoxelCoord(cands, img_dir, npOrigin, npSpacing)
            candNum = len(labels)
            # print(candNum)

            pData = np.memmap(filename=DAT_DATA_Path + "/temp.dat",
                              dtype='float32', mode='w+', shape=(vw[0], vw[1], vw[2], candNum))
            plabel = np.memmap(filename=DAT_DATA_Path + "/temp.lbl",
                               dtype='uint8', mode='w+', shape=(1, 1, candNum))
            # print(plabel.shape)

            cNum = 0
            for i, cand in enumerate(voxelCands):
                arg_arange = [int(cand[0] - vw[0] / 2), int(cand[0] + vw[0] / 2), int(cand[1] - vw[1] / 2),
                              int(cand[1] + vw[1] / 2), int(cand[2] - vw[2] / 2), int(cand[2] + vw[2] / 2)]

                # def TP_voxel_extraction(image, arange, voxelWidth):
                #     # chg_scale = [0, 0, 0, 0, 0, 0]
                #     if arange[0] <= 0:
                #         scale = abs(1 - arange[0])
                #         arange[0] = arange[0] + scale
                #         arange[1] = arange[0] + voxelWidth[0]
                #         # print(scale, 'arg0-1')
                #         # chg_scale[0] = chg_scale[0] + scale
                #     if arange[2] <= 0:
                #         scale = abs(1 - arange[2])
                #         arange[2] = arange[2] + scale
                #         arange[3] = arange[2] + voxelWidth[1]
                #         # print(scale, 'arg2-3')
                #         # chg_scale[1] = chg_scale[1] + scale
                #     if arange[4] <= 0:
                #         scale = abs(1 - arange[4])
                #         arange[4] = arange[4] + scale
                #         arange[5] = arange[4] + voxelWidth[2]
                #         # print(scale, 'arg4-5')
                #         # chg_scale[2] = chg_scale[2] + scale
                #
                #     if arange[1] > image.shape[0]:
                #         scale = abs(image.shape[0] - arange[1])
                #         arange[1] = arange[1] - scale
                #         arange[0] = arange[1] - voxelWidth[0]
                #         # print(scale, 'arg1-0')
                #         # chg_scale[3] = chg_scale[3] + scale
                #     if arange[3] > image.shape[1]:
                #         scale = abs(image.shape[1] - arange[3])
                #         arange[3] = arange[3] - scale
                #         arange[2] = arange[3] - voxelWidth[1]
                #         # print(scale, 'arg3-2')
                #         # chg_scale[4] = chg_scale[4] + scale
                #     if arange[5] > image.shape[2]:
                #         scale = abs(image.shape[2] - arange[5])
                #         arange[5] = arange[5] - scale
                #         arange[4] = arange[5] - voxelWidth[2]
                #         # print(scale, 'arg5-4')
                #         # chg_scale[5] = chg_scale[5] + scale
                #
                #     voxelTensor = np.array(image[arange[0]:arange[1], arange[2]:arange[3], arange[4]:arange[5]])
                #     return voxelTensor

                patch = TP_voxel_extraction(Img=normImage, arg=arg_arange, voxelWidth=vw)

                if patch.shape[0] != vw[0] or patch.shape[1] != vw[1] or patch.shape[2] != vw[2]:
                    patch = np.resize(patch, (vw[0], vw[1], vw[2]))

                pData[:, :, :, cNum] = patch.copy()
                plabel[:, :, cNum] = labels[i]
                cNum += 1
                print("voxelWidth: %d, imageNum: %d, candNum: %d, label: %d \n" % (v, ct, i, labels[i]))

            print(cNum)
            pTempData = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".dat",
                                  dtype='float32', mode='w+', shape=(vw[0], vw[1], vw[2], cNum))
            pTemplabel = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".lbl",
                                   dtype='uint8', mode='w+', shape=(1, 1, cNum))

            pTempData[:, :, :, :] = pData[:, :, :, :cNum]
            pTemplabel[:, :, :] = plabel[:, :, :cNum]
            del pData, plabel, pTempData, pTemplabel

            if vw[3] == 'btm':
                btm_SHApes.append([img_dir[5:-4], cNum])
            elif vw[3] == 'mid':
                mid_SHApes.append([img_dir[5:-4], cNum])
            elif vw[3] == 'top':
                top_SHApes.append([img_dir[5:-4], cNum])
                # if v == 0:
                #     plt.hist(btm_count, fignum=1)
                # elif v == 1:
                #     plt.hist(mid_count, fignum=2)
                # elif v == 2:
                #     plt.hist(top_count, fignum=3)
                # plt.xlabel("Change Scale")
                # plt.ylabel("Frequency")
                # plt.show()

    return btm_SHApes, mid_SHApes, top_SHApes

def mk_train_dataset(total_dataset, train_dataset, test_dataset, dataset, originset, model_num):

    model_exec = train_model.model_execute(train_dataset=train_dataset, test_dataset=test_dataset,
                                            dataset=dataset, originset=originset)
    if model_num == 0:
        model_data_path = model_exec.data_path + '/btm'
        flag_data_path = './output_test/btm/btm_train.dat'

        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_origial_CNN(dataset=total_dataset, lists_data=model_exec.origin_dataset,
                                            dataset_name="total", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=test_dataset, lists_data=model_exec.origin_dataset,
                                            dataset_name="test", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="train", data_path=model_data_path, model_num=model_num)

    elif model_num == 1:
        model_data_path = model_exec.data_path + '/mid'
        flag_data_path = './output_test/mid/mid_train.dat'
        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_origial_CNN(dataset=total_dataset, lists_data=model_exec.origin_dataset,
                                            dataset_name="total", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=test_dataset, lists_data=model_exec.origin_dataset,
                                            dataset_name="test", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="train", data_path=model_data_path, model_num=model_num)

    elif model_num == 2:
        model_data_path = model_exec.data_path + '/top'
        flag_data_path = './output_test/top/top_trainTotal.dat'

        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_origial_CNN(dataset=total_dataset, lists_data=model_exec.origin_dataset,
                                            dataset_name="total", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=test_dataset, lists_data=model_exec.origin_dataset,
                                            dataset_name="test", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="train", data_path=model_data_path, model_num=model_num)


def run_pm_train_3D(total_dataset, train_dataset, test_dataset, dataset, originset, model_num):

    model_def = CAD_3D_model.model_def()
    model_exec = CAD_3D_model.model_execute(train_dataset=train_dataset, test_dataset=test_dataset,
                                            dataset=dataset, originset=originset)
    if model_num != 0:
        model_data_path = model_exec.data_path + '/top'
        flag_data_path = './output_test/btm/btm_trainTotal.dat'

        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_pm_CNN(dataset=total_dataset, lists_data=model_exec.origin_dataset,
                                       dataset_name="total", data_path=model_exec.origin_path + '/top',
                                       model_num=model_num)

            model_exec.mk_patch_pm_CNN(dataset=total_dataset, lists_data=model_exec.origin_dataset,
                                       dataset_name="total", data_path=model_exec.origin_path+'/top',
                                       model_num=model_num)

            model_exec.mk_patch_pm_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                       dataset_name="train", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_pm_CNN(dataset=test_dataset, lists_data=model_exec.origin_dataset,
                                       dataset_name="test", data_path=model_data_path, model_num=model_num)


        # cross_entropy, softmax, layers, data_node, label_node = model_def.bck_CNN(train=True)
        # cross_entropy, softmax, layers, data_node, label_node = model_def.bck_CNN2(train=True)
        cross_entropy, softmax, layers, data_node, label_node = model_def.bck_3D_CNN(train=True)
        # cross_entropy, softmax, layers, data_node, label_node = model_def.bck_NON_LINEAR_CNN(train=True)
        model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                      label_node=label_node, model_num=3)
        # model_exec.train_PM_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                               label_node=label_node, model_num=model_num)
        # cross_entropy, softmax, layers, data_node, label_node = model_def.bck_CNN(train=False)
        # cross_entropy, softmax, layers, data_node, label_node = model_def.bck_CNN2(train=False)
        cross_entropy, softmax, layers, data_node, label_node = model_def.bck_3D_CNN(train=False)
        # cross_entropy, softmax, layers, data_node, label_node = model_def.bck_NON_LINEAR_CNN(train=False)
        model_exec.test_bck_CNN(softmax=softmax, data_node=data_node, dataset="t", model_epoch=2,
                                start_pos=0, model_num=model_num)

        model_exec.code_test()

        # del cross_entropy, softmax, layers, data_node, label_node

def run_3D_2D_train(total_dataset, train_dataset, test_dataset, dataset, originset):

    model_def_3D = CAD_3D_model.model_def()
    model_exec_3D = CAD_3D_model.model_execute(train_dataset=train_dataset, test_dataset=test_dataset,
                                               dataset=dataset, originset=originset)

    model_def_2D = train_model.model_def()
    model_exec_2D = train_model.model_execute(dataset=dataset, irs_dataset=" ")

    if not os.path.exists('pre_process/output_test/btm_pm/top_trainTotal_reshape.dat'):
        for i, md in enumerate(['btm']):

            model_data_path = model_exec_3D.data_path + '/%s' % md
            model_exec_3D.mk_patch_origial_CNN(dataset=total_dataset, lists_data=originset,
                                               dataset_name="total", data_path=model_data_path, model_num=i)
            if i != 0:
                model_exec_3D.mk_patch_resize(dataset_name='train', model_num=i)

    cross_entropy, softmax, layers, data_node, label_node = model_def_3D.bck_3D_CNN(train=True)

    model_exec_3D.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                     label_node=label_node, model_num=i)

    cross_entropy, softmax, layers, data_node, label_node = model_def_3D.bck_3D_CNN(train=False)

    cross_entropy, softmax, layers, data_node, label_node = model_def_2D.bck_CNN(train=True)

    cross_entropy, softmax, layers, data_node, label_node = model_def_2D.bck_CNN2(train=False)

    model_exec_2D.test_bck_CNN(softmax=softmax, data_node=data_node, dataset="t", model_epoch=2,
                               start_pos=0, model_num=i)

    model_exec_2D.code_test()

def Evaluation_3D_CNN_CSV(result_path, data_size, Scan_lists, model_num):
    if not os.path.exists('submition'):
        os.mkdir('submition')

    cand_path = "CSVFile/candidates.csv"
    cands = readCSV(cand_path)
    # btm_result_path = result_path + "/btm_pm_cnn"
    # mid_result_path = result_path + "/mid_pm_cnn"
    # top_result_path = result_path + "/top_pm_cnn"

    result_shape = (551065, 2)
    if model_num == 0:
        data = np.memmap(filename=result_path + "/btm_pm_cnn.dat", dtype=np.float32, mode="r", shape=result_shape)
        csv_filename = 'submition/sub3D_Multi_level_CNN_btm.csv'
    elif model_num == 1:
        data = np.memmap(filename=result_path + "/mid_pm_cnn.dat", dtype=np.float32, mode="r", shape=result_shape)
        csv_filename = 'submition/sub3D_Multi_level_CNN_mid.csv'
    elif model_num == 2:
        data = np.memmap(filename=result_path + "/top_pm_cnn.dat", dtype=np.float32, mode="r", shape=result_shape)
        csv_filename = 'submition/sub3D_Multi_level_CNN_top.csv'

    # fusion_btm = btm * 0.3
    # fusion_mid = mid * 0.4
    # fusion_top = top * 0.3
    print(data[0], data[1])
    # fusion_p_nd = fusion_btm + fusion_mid + fusion_top

    rows = zip(cands[1:], data[:, 0])
    # rows.insert(["seriesuid,coordX,coordY,coordZ,probability"])
    # line = ["seriesuid,coordX,coordY,coordZ,probability"]
    for row in rows:

        csvTools.writeCSV(filename=csv_filename, lines=row)

        # print("%d: line Done!!" % (c))
        # csv_filename.close()

def Total_Evaluation_3D_CNN_CSV(result_path, data_size, Scan_lists):
    if not os.path.exists('submition'):
        os.mkdir('submition')

    btm_result_path = result_path + "/btm_pm_cnn"
    mid_result_path = result_path + "/mid_pm_cnn"
    top_result_path = result_path + "/top_pm_cnn"

    # lbl = np.memmap(filename="output_test/btm_totalTotal_reshape.lbl", dtype=np.uint8, mode="r")
    # result_shape = (lbl.shape[0], 2)
    btm = np.memmap(filename=btm_result_path + ".dat", dtype=np.float32, mode="r")
    mid = np.memmap(filename=mid_result_path + ".dat", dtype=np.float32, mode="r")
    top = np.memmap(filename=top_result_path + ".dat", dtype=np.float32, mode="r")

    fusion_btm = btm * 0.3
    fusion_mid = mid * 0.4
    fusion_top = top * 0.3

    fusion_p_nd = fusion_btm + fusion_mid + fusion_top

    csv_filename = 'submition/sub3D_Multi_level_CNN.csv'

    for lineNum, p, nd_info in enumerate(fusion_p_nd, Scan_lists):
        if lineNum == 0:

            line = 'seriesuid,coordX,coordY,coordZ,probability'
            csvTools.writeCSV(filename=csv_filename, lines=line)

        else:

            line = nd_info[0] + ',' + nd_info[1] + ',' + nd_info[2] + ',' + nd_info[3] + ',' + p
            csvTools.writeCSV(filename=csv_filename, lines=line)

def Animation_BOTTOM(path):
    btm_lbl = np.memmap(filename='./output_test/btm_pm_V2/btm_trainTotal.lbl', dtype='uint8', mode='r')
    dat = np.memmap(filename='./output_test/btm_pm_V2/top_trainTotal.dat', dtype='float32', mode='r', shape=(26, 40, 40, btm_lbl.shape[0]))

    fig, ax = plt.subplots(1)

    ax.imshow(dat[13, :, :, 0], cmap='gray')

    rect1 = patches.Rectangle((0, 0), 39, 39, linewidth=2, edgecolor='g', facecolor='none')
    rect2 = patches.Rectangle((5, 5), 29, 29, linewidth=2, edgecolor='b', facecolor='none')
    rect3 = patches.Rectangle((10, 10), 19, 19, linewidth=2, edgecolor='y', facecolor='none')

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    f1 = patches.Rectangle((1, 1), 6, 6, linewidth=1, edgecolor='r', facecolor='none')
    f2 = patches.Rectangle((6, 6), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    f3 = patches.Rectangle((11, 11), 11, 11, linewidth=1, edgecolor='r', facecolor='none')

    def init():
        ax.add_patch(f1)
        ax.add_patch(f2)
        ax.add_patch(f3)
        return f1, f2, f3,

    def update_btm(i):
        f1_position = [[1, 1], [17, 1], [32, 1],
                       [1, 17], [17, 17], [32, 17],
                       [1, 32], [17, 32], [32, 32]]
        f2_position = [[6, 6], [14, 6], [23, 6],
                       [6, 14], [14, 14], [23, 14],
                       [6, 23], [14, 23], [23, 23]]
        f3_position = [[11, 11], [15, 11], [17, 11],
                       [11, 15], [15, 15], [17, 15],
                       [11, 17], [17, 17], [17, 17]]

        f1.set_xy(f1_position[i])
        f2.set_xy(f2_position[i])
        f3.set_xy(f3_position[i])

        return f1, f2, f3

    # def update_top(i, f1_Num, f2_Num, f3_Num):
    #     f3_Num += i
    #     f2_Num += f3_Num + 2
    #     f1_Num += f2_Num + 2
    #
    #     f1 = patches.Rectangle((0 + f1_Num, 0 + f1_Num), 14, 14, linewidth=1, edgecolor='r', facecolor='none')
    #     f2 = patches.Rectangle((5 + f2_Num, 5 + f2_Num), 7, 7, linewidth=1, edgecolor='r', facecolor='none')
    #     f3 = patches.Rectangle((10 + f3_Num, 10 + f3_Num), 3, 3, linewidth=1, edgecolor='r', facecolor='none')
    #
    #     ax.add_patch(f1)
    #     ax.add_patch(f2)
    #     ax.add_patch(f3)
    #     return ax

    anim = FuncAnimation(fig, update_btm, init_func=init, frames=np.arange(0, 9), interval=200)

    # FFwriter = animation.FFMpegWriter()
    anim.save('BottomUP.mp4', writer='ffmpeg', dpi=80)

def Animation_TOP(path):
    btm_lbl = np.memmap(filename='./output_test/btm_pm_V2/btm_trainTotal.lbl', dtype='uint8', mode='r')
    dat = np.memmap(filename='./output_test/btm_pm_V2/top_trainTotal.dat', dtype='float32', mode='r', shape=(26, 40, 40, btm_lbl.shape[0]))

    fig, ax = plt.subplots(1)

    ax.imshow(dat[13, :, :, 0], cmap='gray')

    rect1 = patches.Rectangle((0, 0), 39, 39, linewidth=2, edgecolor='g', facecolor='none')
    rect2 = patches.Rectangle((5, 5), 29, 29, linewidth=2, edgecolor='b', facecolor='none')
    rect3 = patches.Rectangle((10, 10), 19, 19, linewidth=2, edgecolor='y', facecolor='none')

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    f1 = patches.Rectangle((1, 1), 22, 22, linewidth=1, edgecolor='r', facecolor='none')
    f2 = patches.Rectangle((6, 6), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
    f3 = patches.Rectangle((11, 11), 3, 3, linewidth=1, edgecolor='r', facecolor='none')

    def init():
        ax.add_patch(f1)
        ax.add_patch(f2)
        ax.add_patch(f3)
        return f1, f2, f3,

    def update_top(i):
        f1_position = [[1, 1],[8, 1], [16, 1],
                       [1, 8], [8, 8], [16, 8],
                       [1, 16], [8, 16], [16, 16]]
        f2_position = [[6, 6], [14, 6], [23, 6],
                       [6, 14], [14, 14], [23, 14],
                       [6, 23], [14, 23], [23, 23]]
        f3_position = [[11, 11],[17, 11], [25, 11],
                       [11, 17], [17, 17], [25, 17],
                       [11, 25],[17, 25],[25, 25]]

        f1.set_xy(f1_position[i])
        f2.set_xy(f2_position[i])
        f3.set_xy(f3_position[i])

        return f1, f2, f3

    # def update_top(i, f1_Num, f2_Num, f3_Num):
    #     f3_Num += i
    #     f2_Num += f3_Num + 2
    #     f1_Num += f2_Num + 2
    #
    #     f1 = patches.Rectangle((0 + f1_Num, 0 + f1_Num), 14, 14, linewidth=1, edgecolor='r', facecolor='none')
    #     f2 = patches.Rectangle((5 + f2_Num, 5 + f2_Num), 7, 7, linewidth=1, edgecolor='r', facecolor='none')
    #     f3 = patches.Rectangle((10 + f3_Num, 10 + f3_Num), 3, 3, linewidth=1, edgecolor='r', facecolor='none')
    #
    #     ax.add_patch(f1)
    #     ax.add_patch(f2)
    #     ax.add_patch(f3)
    #     return ax

    anim = FuncAnimation(fig, update_top, init_func=init, frames=np.arange(0, 9), interval=200)

    # FFwriter = animation.FFMpegWriter()
    anim.save('TopDown.mp4', writer='ffmpeg', dpi=80)

def FROC_curve():
    topdown = readCSV(filename='./evaluationScript/Submission/fold_CANDV2/froc_CPX_V2_TOPDOWN.csv')
    bottomup = readCSV(filename='./evaluationScript/Submission/fold_CANDV2/froc_CPX_V2_BOTTOMUP.csv')
    cpx = readCSV(filename='./evaluationScript/Submission/fold_CRM_RESMODULE/froc_CRM_CNN_V2_RES.csv')

    bPerformBootstrapping = True
    bNumberOfBootstrapSamples = 1000
    bOtherNodulesAsIrrelevant = True
    bConfidence = 0.95

    seriesuid_label = 'seriesuid'
    coordX_label = 'coordX'
    coordY_label = 'coordY'
    coordZ_label = 'coordZ'
    diameter_mm_label = 'diameter_mm'
    CADProbability_label = 'probability'

    # plot settings
    FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
    FROC_maxX = 8  # Maximum value of x-axis of FROC curve
    bLogPlot = True

    sens_td_mean, sens_bu_mean, sens_cpx_mean = [], [], []
    sens_td_lb, sens_bu_lb, sens_cpx_lb = [], [], []
    sens_td_up, sens_bu_up, sens_cpx_up = [], [], []

    for t in topdown:
        sens_td_lb.append(t[2])
        sens_td_mean.append(t[1])
        sens_td_up.append(t[3])

    for  b in bottomup:
        sens_bu_lb.append(b[2])
        sens_bu_mean.append(b[1])
        sens_bu_up.append(b[3])

    for c in cpx:
        sens_cpx_lb.append(c[2])
        sens_cpx_mean.append(c[1])
        sens_cpx_up.append(c[3])

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10000)
    fig1 = plt.figure()
    ax = plt.gca()


    # plt.plot(fps_itp, sens_td_mean, color='b', ls='-', label='Zoom-in')
    # plt.plot(fps_itp, sens_td_lb, color='b', ls=':')  # , label = "lb")
    # plt.plot(fps_itp, sens_td_up, color='b', ls=':')  # , label = "ub")
    # ax.fill_between(fps_itp, sens_td_lb, sens_td_up, facecolor='b', alpha=0.05)

    # plt.plot(fps_itp, sens_bu_mean, color='y', ls='-', label='Zoom-out')
    # plt.plot(fps_itp, sens_bu_lb, color='y', ls=':')  # , label = "lb")
    # plt.plot(fps_itp, sens_bu_up, color='y', ls=':')  # , label = "ub")
    # ax.fill_between(fps_itp, sens_bu_lb, sens_bu_up, facecolor='y', alpha=0.05)

    plt.plot(fps_itp, sens_cpx_mean, color='r', ls='-', label='Multi-stream')
    plt.plot(fps_itp, sens_cpx_lb, color='r', ls=':')  # , label = "lb")
    plt.plot(fps_itp, sens_cpx_up, color='r', ls=':')  # , label = "ub")
    # ax.fill_between(fps_itp, sens_bu_lb, sens_bu_up, facecolor='y', alpha=0.05)

    xmin = FROC_minX
    xmax = FROC_maxX
    plt.xlim(xmin, xmax)
    plt.ylim(0.5, 1)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    # ax.legend(handles, ['Zoom-in', 'Zoom-out',])
    plt.title('FROC performance')

    ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
    plt.grid(b=True, which='both')
    plt.tight_layout()

    plt.savefig("froc_multi_stream.png")

def make_lungmask(img, display=False):
    """
    # Standardize the pixel value by subtracting the mean and dividing by the standard deviation
    # Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
    # Using Erosion and Dilation which has the net effect of removing tiny features like pulmonary vessels or noise
    # Identify each distinct region as separate image labels (think the magic wand in Photoshop)
    # Using bounding boxes for each image label to identify which ones represent lung and which ones represent "every thing else"
    # Create the masks for lung fields.
    # Apply mask onto the original image to erase voxels outside of the lung fields.
    """
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if display:
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
        plt.savefig('t.png')
    return mask * img

def resize3D(data, resize, model, flag):
    resizing = np.memmap(filename=model, dtype=np.float32, mode="w+", shape=resize)
    # if  flag == 0:
    #     resizing[:, :, :, :, :, :] = data[10:-10, 10:-10, 10:-10, :, :, :]
    # el
    if flag == 0:
        zSize = 6 / data.shape[3]
        xSize = 20 / data.shape[4]
        ySize = 20 / data.shape[5]

        data = data[8:-8, 5:-5, 5:-5, :, :, :]

        for z in range(resize[0]):
            for x in range(resize[1]):
                for y in range(resize[2]):
                    resizing[z, x, y, :, :, :] = nd.interpolation.zoom(data[z, x, y, :, :, :], zoom=(zSize, xSize, ySize))
                    print('z: %d, x: %d, y: %d' % (z,x,y))
    elif flag == 1:
        zSize = 6 / data.shape[3]
        xSize = 20 / data.shape[4]
        ySize = 20 / data.shape[5]

        for z in range(resize[0]):
            for x in range(resize[1]):
                for y in range(resize[2]):
                    resizing[z, x, y, :, :, :] = nd.interpolation.zoom(data[z, x, y, :, :, :], zoom=(zSize, xSize, ySize))
                    print('z: %d, x: %d, y: %d' % (z,x,y))

    del resizing
