
import SimpleITK as sitk
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import misc
from random import shuffle
from PIL import Image
from revised_code import train_model

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    numpyImage = np.transpose(numpyImage, axes=(1,2,0))
    numpyOrigin = [numpyOrigin[1], numpyOrigin[2], numpyOrigin[0]]
    numpySpacing = [numpySpacing[1], numpySpacing[2], numpySpacing[0]]
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

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

# def img_rotation(img):



def chg_VoxelCoord(lists, str, origin, spacing):
    cand_list = []
    labels = []
    for list in lists:
        if list[0] in str:
            worldCoord = np.asarray([float(list[2]), float(list[1]), float(list[3])])
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
                  [0, 0, 1], [0, 0, 0], [0, 0, -1],
                  [0, -1, 1], [0, -1, 0], [0, -1, -1],
                  [-1, 1, 1], [-1, 1, 0], [-1, 1, -1],
                  [-1, 0, 1], [-1, 0, 0], [-1, 0, -1],
                  [-1, -1, 1], [-1, -1, 0], [1, -1, -1]]

    aug = list + shiftCoord
    aug_size = len(aug)
    return aug, aug_size

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    #for i in range(k):
    validation = slices[0]
    test = slices[1]
    training = slices[2] + slices[3] + slices[4]
    return training, validation, test

def nodule_patch_extraction(DAT_DATA_Path, CT_scans):
    cand_path = "CSVFile/candidates.csv"
    cands = readCSV(cand_path)
    totalSHApes = []
    for img_dir in CT_scans:
        npImage, npOrigin, npSpacing = load_itk_image(img_dir)
        normImage = normalizePlanes(npImage)
        print(normImage.shape)
        voxelCands, labels = chg_VoxelCoord(cands, img_dir, npOrigin, npSpacing)
        candNum = len(labels)
        voxelWidth = [[20, 20, 6, ['btm']], [30, 30, 10, ['mid']], [40, 40, 26, ['top']]]
        for v in voxelWidth:
            pData = np.memmap(filename=DAT_DATA_Path + "/temp.dat",
                                    dtype='float32', mode='w+', shape=(v[0], v[1], v[2], candNum))
            plabel = np.memmap(filename=DAT_DATA_Path + "/temp.lbl",
                                dtype='uint8', mode='w+', shape=(1, 1, candNum))
            cNum = 0
            for i, cand in enumerate(voxelCands):
                arg_arange = [int(cand[0] - v[0] / 2), int(cand[0] + v[0] / 2), int(cand[1] - v[1] / 2),
                              int(cand[1] + v[1] / 2), int(cand[2] - v[2] / 2), int(cand[2] + v[2] / 2)]
                cand_min_scale = (int(arg_arange[0] > normImage.shape[0]) + int(arg_arange[2] > normImage.shape[1])
                                  + int(arg_arange[4] > normImage.shape[2]))
                cand_max_scale = (int(arg_arange[1] > normImage.shape[0]) + int(arg_arange[3] > normImage.shape[1])
                                  + int(arg_arange[5] > normImage.shape[2]))
                if not labels[i] == '0':
                    if not cand_min_scale > 0 or not cand_max_scale > 0:
                        patch = np.array(normImage[arg_arange[0]:arg_arange[1], arg_arange[2]:arg_arange[3], arg_arange[4]:arg_arange[5]])
                        # print(patch.shape)
                        if not patch.shape[2] == v[2]:
                            if normImage.shape[2] < arg_arange[5]:
                                scale = normImage.shape[2] - arg_arange[5]
                            elif arg_arange[4] == 0:
                                scale = arg_arange[4] + 1
                            patch = np.array(normImage[arg_arange[0]:arg_arange[1], arg_arange[2]:arg_arange[3],
                                             (arg_arange[4] + scale):(arg_arange[5] + scale)])
                            print(patch.shape, 'change')
                        elif not patch.shape[1] == v[1]:
                            if normImage.shape[1] < arg_arange[3]:
                                scale = normImage.shape[1] - arg_arange[3]
                            elif arg_arange[2] == 0:
                                scale = arg_arange[2] + 1
                            patch = np.array(normImage[arg_arange[0]:arg_arange[1],
                                             (arg_arange[2] + scale):(arg_arange[3] + scale),
                                             arg_arange[4]:arg_arange[5]])
                            print(patch.shape, 'change')
                        elif not patch.shape[0] == v[0]:
                            if normImage.shape[0] < arg_arange[1]:
                                scale = normImage.shape[0] - arg_arange[1]
                            elif arg_arange[0] == 0:
                                scale = arg_arange[0] + 1
                            patch = np.array(normImage[(arg_arange[0] + scale):(arg_arange[1] + scale),
                                             arg_arange[2]:arg_arange[3],
                                             arg_arange[4]:arg_arange[5]])
                            print(patch.shape, 'change')
                        # plt.matshow(patch[:, :, 3], fignum=1, cmap=plt.cm.gray)
                        pData[:, :, :, cNum] = patch.copy()
                        plabel[:, :, cNum] = labels[i]
                        cNum += 1
                    else:
                        print(cand)

        pTempData = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + str(v[3]) + ".dat",
                          dtype='float32', mode='w+', shape=(v[0], v[1], v[2], cNum))
        pTemplabel = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".lbl",
                           dtype='uint8', mode='w+', shape=(1, 1, cNum))

        pTempData[:, :, :, :] = pData[:, :, :, :cNum]
        pTemplabel[:, :, :]   = plabel[:, :, :cNum]

        del pData, plabel, pTempData, pTemplabel
        totalSHApes.append([img_dir[5:-4], cNum])
    return totalSHApes

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
