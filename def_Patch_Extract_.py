import csv
import os
import math
from random import shuffle

import SimpleITK as sitk
import numpy as np
from revised_code import train_model
from revised_code import CAD_3D_model
import matplotlib.pyplot as plt
import evaluationScript as escript
from evaluationScript.tools import csvTools
from scipy import ndimage as nd


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


def cal_diameter(target, candidate, diameter):
    # dist = sum([ math.sqrt((abs(float(t))-abs(float(c)))**2) for t, c in zip(target, candidate) ])
    dist = [ ((float(t) - float(c)) ** 2) for t, c in zip(target, candidate)]
    dist = math.sqrt(sum(dist))
    if dist <= float(diameter):
        return True
    else:
        return False


def resample(image, spacing, new_spacing):

    resize_factor = np.array(spacing) / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    extract_voxel_size = spacing / real_resize_factor

    return extract_voxel_size


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


def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a: object, window: object) -> object:
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a


def img_rotation(img, k):

    rot_img = np.rot90(img, k, axes=(1,2))

    return rot_img



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
    # else:
    #     for list in lists:
    #         if list[0] in str:
    #             worldCoord = np.asarray([float(list[1]), float(list[2]), float(list[3])])
    #             voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
    #             augs, aug_labels = aug_candidate(voxelCoord)
    #             cand_list.append(voxelCoord)
    #             labels.append(1)
    #             for aug in augs:
    #                 cand_list.append(aug)
    #             al_vec = np.ones((int(aug_labels), 1))
    #             for aug_lbl in al_vec:
    #                 labels.append(int(aug_lbl))
    #
    #     return cand_list, labels



def VoxelCoord(lists, str, origin, spacing):
    cand_list = []

    for list in lists:
        if list[0] in str:
            worldCoord = np.asarray([float(list[1]), float(list[2]), float(list[3])])
            voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
            diameter = math.floor(float(list[4])/spacing[0])
            cand_list.append([voxelCoord[0], voxelCoord[1], voxelCoord[2], diameter])
            print('|', worldCoord, '|->|', voxelCoord, '|')

    return cand_list



def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord



def aug_candidate(list):
    #aug = []
    shiftCoord = [[1, 1, 1], [1, 1, 0], [1, 1, -1],
                  [1, 0, 1], [1, 0, 0], [1, 0, -1],
                  [1, -1, 1], [1, -1, 0], [1, -1, -1],
                  [0, 1, 1], [0, 1, 0], [0, 1, -1],
                  [0, 0, 1], [0, 0, -1], [0, -1, 1],
                  [0, -1, 0], [0, -1, -1], [-1, 1, 1],
                  [-1, 1, 0], [-1, 1, -1], [-1, 0, 1],
                  [-1, 0, 0], [-1, 0, -1], [-1, -1, 1],
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
    training = slices[0] + slices[2] + slices[3] + slices[4]
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



def nodule_patch_extraction(DAT_DATA_Path, CT_scans):
    cands = readCSV(filename="CSVFile/candidates.csv")
    # annot = readCSV(filename="CSVFile/annotations.csv")

    btm_SHApes = []
    mid_SHApes = []
    top_SHApes = []

    DAT_Path = DAT_DATA_Path
    voxelWidth = [[6, 20, 20, 'btm'], [10, 30, 30, 'mid'], [26, 40, 40, 'top']]
    # voxelWidth = [[6, 20, 20, 'btm']]

    for v, vw in enumerate(voxelWidth):

        DAT_DATA_Path = DAT_Path
        DAT_DATA_Path = DAT_DATA_Path + '/%s' % (vw[3])
        # f = open(DAT_DATA_Path + '_' + "patient_list.txt", 'w')
        VW_std = vw
        if not os.path.exists(DAT_DATA_Path):
            os.mkdir(DAT_DATA_Path)
        print(DAT_DATA_Path)
        for ct, img_dir in enumerate(CT_scans):
            npImage, npOrigin, npSpacing = load_itk_image(img_dir)
            normImage = normalizePlanes(npImage)
            # print(normImage.shape)
            voxelCands, labels = chg_VoxelCoord(cands, img_dir, npOrigin, npSpacing)

            candNum = labels.count(0) + (labels.count(1) * 4)

            pData = np.memmap(filename=DAT_DATA_Path + "/temp.dat",
                              dtype='float32', mode='w+', shape=(VW_std[0], VW_std[1], VW_std[2], candNum))
            plabel = np.memmap(filename=DAT_DATA_Path + "/temp.lbl",
                               dtype='uint8', mode='w+', shape=(1, 1, candNum))
            cNum = 0
            for i, cand in enumerate(voxelCands):
                arg_arange = [(math.floor(cand[0]) - vw[0] / 2), (math.floor(cand[0]) + vw[0] / 2),
                              (math.floor(cand[1]) - vw[1] / 2), (math.floor(cand[1]) + vw[1] / 2),
                              (math.floor(cand[2]) - vw[2] / 2), (math.floor(cand[2]) + vw[2] / 2)]

                patch = TP_voxel_extraction(Img=normImage, arg=arg_arange, voxelWidth=vw)
                # plt.matshow(patch[:, :, 3], fignum=1, cmap=plt.cm.gray)
                # plt.matshow(normImage[:, :, int(cand[2])], fignum=2, cmap=plt.cm.gray)
                # plt.plot(int(cand[0]), int(cand[1]), 'r+')
                # plt.axis([511, 0, 511], fignum=2)
                # plt.show()

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
                                  dtype='float32', mode='w+', shape=(VW_std[0], VW_std[1], VW_std[2], cNum))
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



def slide_extraction(DAT_DATA_Path, CT_scans):
    # cands = readCSV(filename="CSVFile/candidates.csv")
    annot = readCSV(filename="CSVFile/annotations.csv")

    DAT_DATA_Path = DAT_DATA_Path + '/Slide'

    if not os.path.exists(DAT_DATA_Path):
        os.mkdir(DAT_DATA_Path)

    print(DAT_DATA_Path)
    for ct, img_dir in enumerate(CT_scans):
        npImage, npOrigin, npSpacing = load_itk_image(img_dir)
        normImage = normalizePlanes(npImage)

        reSizing = [512, 512, 200]
        tempImage = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".dat",
                              dtype='float32', mode='w+', shape=(reSizing))
        for x in range(0, 512):
            for y in range(0, 512):
                tempImage[x, y, :] = nd.zoom(normImage[x, y, :], 200 / normImage.shape[2], order=0)

        print('|', normImage.shape, '|->|', tempImage.shape, '|')

        voxelAnnot = VoxelCoord(annot, img_dir, npOrigin, npSpacing)

        for va in voxelAnnot:
            minX = int(va[0] - (va[3] / 2))
            maxX = int(va[0] + (va[3] / 2))
            minY = int(va[1] - (va[3] / 2))
            maxY = int(va[1] + (va[3] / 2))
            minZ = int(va[2] - (va[3] / 2))
            maxZ = int(va[2] + (va[3] / 2))

            tempImage[minY:maxY, minX:maxX, minZ:maxZ] = 1


        print(tempImage.shape)

        plabel = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".lbl",
                           dtype='uint8', mode='w+', shape=(reSizing))
        plabel = tempImage.copy



def slide_label_extraction(DAT_DATA_Path, CT_scans):
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



def total_patch_extraction(DAT_DATA_Path, CT_scans):
    cand_path = "CSVFile/candidates.csv"
    cands = readCSV(cand_path)

    btm_SHApes = []
    mid_SHApes = []
    top_SHApes = []

    voxelWidth = [[20, 20, 6, 'btm'], [30, 30, 10, 'mid'], [40, 40, 26, 'top']]
    VWidth_std = [30, 30, 10]

    for v, vw in enumerate(voxelWidth):
        DAT_DATA_Path = 'origin_ext_voxel'
        if not os.path.exists(DAT_DATA_Path):
            os.mkdir(DAT_DATA_Path)

        DAT_DATA_Path = DAT_DATA_Path + '/%s' % (vw[3])
        if not os.path.exists(DAT_DATA_Path):
            os.mkdir(DAT_DATA_Path)

        print(DAT_DATA_Path)
        for ct, img_dir in enumerate(CT_scans):
            npImg, npOrigin, npSpacing = load_itk_image(img_dir)
            normImage = normalizePlanes(npImage)
            print(normImage.shape)
            voxelCands, labels = org_VoxelCoord(cands, img_dir, npOrigin, npSpacing)
            candNum = len(labels)
            print(candNum)

            pData = np.memmap(filename=DAT_DATA_Path + "/temp.dat",
                              dtype='float32', mode='w+', shape=(VWidth_std[0], VWidth_std[1], VWidth_std[2], candNum))
            plabel = np.memmap(filename=DAT_DATA_Path + "/temp.lbl",
                               dtype='uint8', mode='w+', shape=(1, 1, candNum))
            print(plabel.shape)

            cNum = 0
            for i, cand in enumerate(voxelCands):
                arg_arange = [int(cand[0] - vw[0] / 2), int(cand[0] + vw[0] / 2), int(cand[1] - vw[1] / 2),
                              int(cand[1] + vw[1] / 2), int(cand[2] - vw[2] / 2), int(cand[2] + vw[2] / 2)]

                def TP_voxel_extraction(image, arange, voxelWidth):
                    # chg_scale = [0, 0, 0, 0, 0, 0]
                    if arange[0] <= 0:
                        scale = abs(1 - arange[0])
                        arange[0] = arange[0] + scale
                        arange[1] = arange[0] + voxelWidth[0]
                        # print(scale, 'arg0-1')
                        # chg_scale[0] = chg_scale[0] + scale
                    if arange[2] <= 0:
                        scale = abs(1 - arange[2])
                        arange[2] = arange[2] + scale
                        arange[3] = arange[2] + voxelWidth[1]
                        # print(scale, 'arg2-3')
                        # chg_scale[1] = chg_scale[1] + scale
                    if arange[4] <= 0:
                        scale = abs(1 - arange[4])
                        arange[4] = arange[4] + scale
                        arange[5] = arange[4] + voxelWidth[2]
                        # print(scale, 'arg4-5')
                        # chg_scale[2] = chg_scale[2] + scale

                    if arange[1] > image.shape[0]:
                        scale = abs(image.shape[0] - arange[1])
                        arange[1] = arange[1] - scale
                        arange[0] = arange[1] - voxelWidth[0]
                        # print(scale, 'arg1-0')
                        # chg_scale[3] = chg_scale[3] + scale
                    if arange[3] > image.shape[1]:
                        scale = abs(image.shape[1] - arange[3])
                        arange[3] = arange[3] - scale
                        arange[2] = arange[3] - voxelWidth[1]
                        # print(scale, 'arg3-2')
                        # chg_scale[4] = chg_scale[4] + scale
                    if arange[5] > image.shape[2]:
                        scale = abs(image.shape[2] - arange[5])
                        arange[5] = arange[5] - scale
                        arange[4] = arange[5] - voxelWidth[2]
                        # print(scale, 'arg5-4')
                        # chg_scale[5] = chg_scale[5] + scale

                    voxelTensor = np.array(image[arange[0]:arange[1], arange[2]:arange[3], arange[4]:arange[5]])
                    return voxelTensor
                    # if not ((int(patch.shape[0] == vw[0]) + int(patch.shape[1] == vw[1]) +
                    #              int(patch.shape[2] == vw[2])) == 3):
                    #     print((arg_arange[0] > (0 - 4)) and (0 - arg_arange[0]) <= 4)
                    #     print((arg_arange[2] > (0 - 4)) and (0 - arg_arange[2]) <= 4)
                    #     print((arg_arange[4] > (0 - 4)) and (0 - arg_arange[4]) <= 4)
                    #     print((arg_arange[1] <= (normImage.shape[0] + 4)) and (arg_arange[1] - normImage.shape[0]) <= 4)
                    #     print((arg_arange[3] <= (normImage.shape[1] + 4)) and (arg_arange[3] - normImage.shape[1]) <= 4)
                    #     print((arg_arange[5] <= (normImage.shape[2] + 4)) and (arg_arange[5] - normImage.shape[2]) <= 4)
                    #     print(patch.shape)

                patch = TP_voxel_extraction(image=normImage, arange=arg_arange, voxelWidth=vw)

                if not patch.shape[0] == VWidth_std[0]:
                    patch = np.resize(patch, (VWidth_std[0], VWidth_std[1], VWidth_std[2]))

                pData[:, :, :, cNum] = patch.copy()
                plabel[:, :, cNum] = labels[i]
                cNum += 1

                s = "voxelWidth: %d, imageNum: %d, candNum: %d, label: %d \n" % (v, ct, i, labels[i])
                print(s)
            print(cNum)
            pTempData = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".dat",
                                  dtype='float32', mode='w+', shape=(VWidth_std[0], VWidth_std[1], VWidth_std[2], cNum))
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



def run_train(train_dataset, val_dataset, test_dataset, dataset):
    model_def = train_model.model_def()
    # dataset = train or test define / irs_dataset = normalize method of prepocessing % one of hgg / vgg
    model_exec = train_model.model_execute(train_dataset=train_dataset, val_dataset=val_dataset,
                                           test_dataset=test_dataset, dataset=dataset)

    if not os.path.exists('output_test/testTotal.dat'):
        model_exec.extract_patch_original_CNN()

    cross_entropy, softmax, layers, data_node, label_node = model_def.original_CNN(train=True)
    model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                  label_node=label_node)
    cross_entropy, softmax, layers, data_node, label_node = model_def.original_CNN(train=False)
    model_exec.test_original_CNN(softmax=softmax, data_node=data_node, dataset="t", model_epoch=str(10))



def run_train_3D(total_dataset, train_dataset, test_dataset, dataset, originset, model_num):

    model_def = CAD_3D_model.model_def()
    model_exec = CAD_3D_model.model_execute(train_dataset=train_dataset, test_dataset=test_dataset,
                                            dataset=dataset, originset=originset)

    if model_num == 0:
        model_data_path = model_exec.data_path + '/btm'
        flag_data_path = './output_test/btm/btm_trainTotal.dat'

        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="train", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=test_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="test", data_path=model_data_path, model_num=model_num)

        cross_entropy, softmax, layers, data_node, label_node = model_def.btm_CNN(train=True)
        model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                      label_node=label_node, model_num=m)

        cross_entropy, softmax, layers, data_node, label_node = model_def.btm_CNN(train=False)

        model_exec.test_original_CNN(softmax=softmax, data_node=data_node, model_epoch=str(0), model_num=model_num)
        # model_exec.pre_train_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                          label_node=label_node, model_num=m)
        # del cross_entropy, softmax, layers, data_node, label_node

        # cross_entropy, softmax, layers, data_node, label_node = model_def.btm_CNN(train=True)
        # model_exec.fine_tune_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                          label_node=label_node, model_num=m)

        del cross_entropy, softmax, layers, data_node, label_node
    elif model_num == 1:
        model_data_path = model_exec.data_path + '/mid'
        flag_data_path = '/output_test/mid/mid_trainTotal.dat'
        cross_entropy, softmax, layers, data_node, label_node = model_def.mid_CNN(fine_Flag=True, train=True)
        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_origial_CNN(dataset=total_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="fine", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=test_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="test", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="train", data_path=model_data_path, model_num=model_num)

        # model_exec.pre_train_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                          label_node=label_node, model_num=m)
        #
        # cross_entropy, softmax, layers, data_node, label_node = model_def.mid_CNN(fine_Flag=True, train=True)
        # model_exec.fine_tune_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                          label_node=label_node, model_num=m)
        model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                      label_node=label_node, model_num=model_num)
        del cross_entropy, softmax, layers, data_node, label_node
    elif model_num == 2:
        model_data_path = model_exec.data_path + '/top'
        flag_data_path = model_exec.hdd_output_path + 'top/top_trainTotal.dat'

        if not os.path.exists(flag_data_path):
            model_exec.mk_patch_origial_CNN(dataset=test_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="test", data_path=model_data_path, model_num=model_num)

            model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
                                            dataset_name="train", data_path=model_data_path, model_num=model_num)

        cross_entropy, softmax, layers, data_node, label_node = model_def.top_CNN(train=True)
        # model_exec.pre_train_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                          label_node=label_node, model_num=m)
        #
        # cross_entropy, softmax, layers, data_node, label_node = model_def.top_CNN(train=True)
        # model_exec.fine_tune_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
        #                          label_node=label_node, model_num=m)
        model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
                                      label_node=label_node, model_num=model_num)
        del cross_entropy, softmax, layers, data_node, label_node
        # if not os.path.exists(flag_data_path):
        #     model_exec.mk_patch_origial_CNN(dataset=total_dataset, lists_data=model_exec.lists_data,
        #                                     dataset_name="fine", data_path=model_data_path, model_num=m)
        #
        #     model_exec.mk_patch_origial_CNN(dataset=val_dataset, lists_data=model_exec.lists_data,
        #                                     dataset_name="val", data_path=model_data_path, model_num=m)
        #
        #     model_exec.mk_patch_origial_CNN(dataset=train_dataset, lists_data=model_exec.lists_data,
        #                                     dataset_name="train", data_path=model_data_path, model_num=m)

        # model_exec.mk_patch_origial_CNN(dataset=total_dataset, lists_data=model_exec.origin_dataset,
        #                                 dataset_name="total", data_path=model_data_path, model_num=m)

    # model_exec.pre_train_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
    #                         label_node=label_node, model_num=m)
    # del cross_entropy, softmax, layers, data_node, label_node
    #
    # cross_entropy, softmax, layers, data_node, label_node = model_def.btm_CNN(train=True)
    # model_exec.fine_tune_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
    #                          label_node=label_node, model_num=m)

    # model_exec.train_original_CNN(cross_entropy=cross_entropy, softmax=softmax, data_node=data_node,
    #                               label_node=label_node, model_num=m)

    if model_num == 0:
        cross_entropy, softmax, layers, data_node, label_node = model_def.btm_CNN(train=False)
        model_exec.test_original_CNN(softmax=softmax, data_node=data_node, model_epoch=str(27), model_num=model_num)

        Evaluation_3D_CNN_CSV(result_path='output_test', data_size=originset, Scan_lists=total_dataset, model_num=model_num)
    elif model_num == 1:
        cross_entropy, softmax, layers, data_node, label_node = model_def.mid_CNN(fine_Flag=False, train=False)
        model_exec.test_original_CNN(softmax=softmax, data_node=data_node, model_epoch=str(9), model_num=model_num)

        Evaluation_3D_CNN_CSV(result_path='output_test', data_size=originset, Scan_lists=total_dataset, model_num=model_num)
    elif model_num == 2:
        cross_entropy, softmax, layers, data_node, label_node = model_def.top_CNN(train=False)
        model_exec.test_original_CNN(softmax=softmax, data_node=data_node, model_epoch=str(10), model_num=model_num)

        Evaluation_3D_CNN_CSV(result_path='output_test', data_size=originset, Scan_lists=total_dataset, model_num=model_num)


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

