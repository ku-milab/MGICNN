
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
                aug, aug_label = aug_candidate(voxelCoord)
                cand_list.append((voxelCoord))
                cand_list.append(aug[:len(aug)])
                al_vec = np.ones((int(aug_label),1))
                labels.append(int(list[4]))
                labels.append(al_vec)
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
        voxelCands, labels = chg_VoxelCoord(cands, img_dir, npOrigin, npSpacing)
        candNum = len(labels)
        voxelWidth = np.array([[40, 40, 40], [45, 45, 45], [50, 50, 50], [55, 55, 55]]) // npSpacing

        pData_SML = np.memmap(filename=DAT_DATA_Path + "/temp_sml.dat",
                                dtype='float32', mode='w+', shape=(20, 20, 6, candNum))
        pData_MID = np.memmap(filename=DAT_DATA_Path + "/temp_mid.dat",
                                dtype='float32', mode='w+', shape=(30, 30, 10, candNum))
        pData_LAR = np.memmap(filename=DAT_DATA_Path + "/temp_lar.dat",
                                dtype='float32', mode='w+', shape=(40, 40, 26, candNum))
        plabel = np.memmap(filename=DAT_DATA_Path + "/temp.lbl",
                            dtype='uint8', mode='w+', shape=(1, 1, candNum))
        cNum = 0
        for i, cand in enumerate(voxelCands):
            tensor = rolling_window(normImage, (voxelWidth[2]))
            print(tensor.shape)
            SHAapes = tensor.shape
            VCoord = int(cand - (voxelWidth[2] // 2))
            if not (VCoord[0] >= SHAapes[0]) | (VCoord[1] >= SHAapes[1]) | (VCoord[2] >= SHAapes[2]):
                patch = np.array(tensor[VCoord[0], VCoord[1], VCoord[2]])
                # pTemp = misc.imresize(patch, (64, 64), interp='bilinear', mode='F')
                # print(pTemp.shape)
                plt.matshow(patch, fignum=1, cmap=plt.cm.gray)
                # pData[:,:,cNum] = pTemp.copy()
                plabel[:, :, cNum] = labels[i]
                cNum += 1
            else:
                print(cand)
        for gt_cand in shiftCands:
            for cand in gt_cand:
                for vw in voxelWidth:
                    tensors = rolling_window(normImage, (vw, vw))
                    #print(tensor.shape)
                    Shapes = tensors.shape[:]

                    VCord = (int(cand[0] - (vw // 2)), int(cand[1] - (vw // 2)), int(cand[2] - (vw // 2)))
                    #print(VCoord)
                    if not (VCord[0] >= Shapes[0]) | (VCord[1] >= Shapes[1]) | (VCord[2] >= Shapes[2]):
                        patches = np.array(tensors[VCord[0], VCord[1], VCord[2]])
                        # pTempes = misc.imresize(patches, (64, 64), interp='bilinear', mode='F')
                        # plt.matshow(pTemp, fignum=1, cmap=plt.cm.gray)
                        # pData[:,:,cNum] = pTempes.copy()
                        plabel[:, :, cNum] = 1
                        cNum += 1
                    else:
                        print(gt_cand)

        pTempData = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".dat",
                          dtype='float32', mode='w+', shape=(64, 64, cNum))
        pTemplabel = np.memmap(filename=DAT_DATA_Path + "/" + img_dir[5:-4] + ".lbl",
                           dtype='uint8', mode='w+', shape=(1, 1, cNum))

        pTempData[:, :, :] = pData[:, :, :cNum]
        pTemplabel[:, :, :] = plabel[:, :, :cNum]

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
