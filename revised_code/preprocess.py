from __future__ import print_function

import os
import pickle
import time
from datetime import datetime

import medpy.io
import numpy as np
from medpy.filter import IntensityRangeStandardization

HDD_DATA_PATH = "data/"
HDD_OUTPUT_PATH = "output/"
MOD = {"T1": 0, "T2": 1, "T1c": 2, "Flair": 3, "OT": 4}
HL = {"h": 0, "l": 1, "t": 2}
SHAPE = (240, 240, 155, 4)
MOD_CNT = 4
LABEL_CNT = 5


def get_filename(path=HDD_DATA_PATH):
    # h.86.VSD.Brain.XX.O.MR_T1.35779.nii
    name_list = os.listdir(path)
    result = {}
    for name in name_list:
        temp = name.replace(".nii", "").split(".")
        temp_dataset = HL[temp[0]]
        temp_num = int(temp[1]) - 1
        temp_mod = MOD[temp[-2].split("_")[-1]]
        result[temp_dataset, temp_num, temp_mod] = path + name
    return result


def get_img(filename, isOT=False):
    result = medpy.io.load(filename)[0]
    if not isOT:
        minpxl = np.min(result)
        if minpxl < 0:
            result[result != 0] -= minpxl
    return result


def save_orig_data():
    logthis("Original data saving started!")
    filenames = get_filename()
    h_cnt = 0
    l_cnt = 0
    t_cnt = 0
    hl_cnt = 0
    for names in filenames:
        if names[-1] == 0:
            if names[0] == HL["h"]:
                h_cnt += 1
                hl_cnt += 1
            elif names[0] == HL["l"]:
                l_cnt += 1
                hl_cnt += 1
            elif names[0] == HL["t"]:
                t_cnt += 1

    # h_data = np.memmap(filename=HDD_OUTPUT_PATH + "h_orig.dat", dtype=np.float32, mode="w+",
    #                    shape=(h_cnt, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]))
    # h_label = np.memmap(filename=HDD_OUTPUT_PATH + "h_orig.lbl", dtype=np.uint8, mode="w+",
    #                     shape=(h_cnt, SHAPE[0], SHAPE[1], SHAPE[2]))

    hl_data = np.memmap(filename=HDD_OUTPUT_PATH + "hl_orig.dat", dtype=np.float32, mode="w+",
                        shape=(hl_cnt, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]))
    hl_label = np.memmap(filename=HDD_OUTPUT_PATH + "hl_orig.lbl", dtype=np.uint8, mode="w+",
                         shape=(hl_cnt, SHAPE[0], SHAPE[1], SHAPE[2]))

    for curcnt in range(h_cnt):
        local_time = time.time()
        cur_lbl = get_img(filenames[HL["h"], curcnt, MOD["OT"]], isOT=True)
        hl_label[curcnt] = cur_lbl
        for mods in range(MOD_CNT):
            cur_data = get_img(filenames[HL["h"], curcnt, mods])
            # h_data[curcnt, ..., mods] = cur_data
            # h_label[curcnt] = cur_lbl
            hl_data[curcnt, ..., mods] = cur_data
        print("\rOriginal data saving (h)", curcnt + 1, "/", h_cnt, time.time() - local_time, end="")

    # l_data = np.memmap(filename=HDD_OUTPUT_PATH + "l_orig.dat", dtype=np.float32, mode="w+",
    #                    shape=(l_cnt, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]))
    # l_label = np.memmap(filename=HDD_OUTPUT_PATH + "l_orig.lbl", dtype=np.uint8, mode="w+",
    #                     shape=(l_cnt, SHAPE[0], SHAPE[1], SHAPE[2]))

    for curcnt in range(l_cnt):
        cur_lbl = get_img(filenames[HL["l"], curcnt, MOD["OT"]], isOT=True)
        hl_label[h_cnt + curcnt] = cur_lbl
        local_time = time.time()
        for mods in range(MOD_CNT):
            cur_data = get_img(filenames[HL["l"], curcnt, mods])
            # l_data[curcnt, ..., mods] = cur_data
            # l_label[curcnt] = cur_lbl
            hl_data[h_cnt + curcnt, ..., mods] = cur_data
        print("\rOriginal data saving (l)", h_cnt + curcnt + 1, "/", hl_cnt, time.time() - local_time, end="")
    t_data = np.memmap(filename=HDD_OUTPUT_PATH + "t_orig.dat", dtype=np.float32, mode="w+",
                       shape=(t_cnt, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]))

    for curcnt in range(t_cnt):
        local_time = time.time()
        for mods in range(MOD_CNT):
            cur_data = get_img(filenames[HL["t"], curcnt, mods])
            t_data[curcnt, ..., mods] = cur_data
        print("\rOriginal data saving (t)", curcnt + 1, "/", t_cnt, time.time() - local_time, end="")


def train_IRS():
    hl_data = np.memmap(filename=HDD_OUTPUT_PATH + "hl_orig.dat", dtype=np.float32, mode="r").reshape(-1, SHAPE[0],
                                                                                                      SHAPE[1],
                                                                                                      SHAPE[2],
                                                                                                      SHAPE[3])

    logthis("HL IRS training started!")
    irs = IntensityRangeStandardization()
    for cur_cnt in range(hl_data.shape[0]):
        for mod_cnt in range(MOD_CNT):
            curmod = hl_data[cur_cnt, ..., mod_cnt]
            irs = irs.train([curmod[curmod > 0]])
        print("\rHL", cur_cnt, end="")
    with open(os.path.join(HDD_OUTPUT_PATH, "hl_irs.dat"), 'wb') as f:
        pickle.dump(irs, f)
    logthis("HL IRS training ended!")


def set_IRS(irs_dataset):
    with open(HDD_OUTPUT_PATH + irs_dataset + "_irs.dat", 'r') as f:
        trained_model = pickle.load(f)

    logthis("T IRS setting started!")
    t_data = np.memmap(filename=HDD_OUTPUT_PATH + "t_orig.dat", dtype=np.float32, mode="r").reshape(-1, SHAPE[0],
                                                                                                    SHAPE[1], SHAPE[2],
                                                                                                    SHAPE[3])
    t_irs_data = np.memmap(filename=HDD_OUTPUT_PATH + "t_%s_irs.dat" % (irs_dataset), dtype=np.float32, mode="w+",
                           shape=t_data.shape)
    for data, new_data, cur_cnt in zip(t_data, t_irs_data, range(t_data.shape[0])):
        for mod_cnt in range(MOD_CNT):
            mod_data = data[..., mod_cnt]
            # temp_calc = mod_data[mod_data>0]-np.min(mod_data[mod_data>0])
            # temp_calc = (temp_calc*(trained_model[mod_cnt].stdrange[-1]-trained_model[mod_cnt].stdrange[0])/np.max(temp_calc))+trained_model[mod_cnt].stdrange[0]
            mod_new_data = new_data[..., mod_cnt]
            mod_new_data[mod_data > 0] = trained_model.transform(mod_data[mod_data > 0],
                                                                 surpress_mapping_check=True)
            mod_new_data[mod_data == 0] = 0
        print("\rT IRS setting", cur_cnt, end="")
    logthis("T IRS setting finished!")

    logthis("HL IRS setting started!")
    hl_data = np.memmap(filename=HDD_OUTPUT_PATH + "hl_orig.dat", dtype=np.float32, mode="r").reshape(-1, SHAPE[0],
                                                                                                      SHAPE[1],
                                                                                                      SHAPE[2],
                                                                                                      SHAPE[3])
    hl_irs_data = np.memmap(filename=HDD_OUTPUT_PATH + "hl_%s_irs.dat" % (irs_dataset), dtype=np.float32, mode="w+",
                            shape=hl_data.shape)
    for data, new_data, cur_cnt in zip(hl_data, hl_irs_data, range(hl_data.shape[0])):
        for mod_cnt in range(MOD_CNT):
            mod_data = data[..., mod_cnt]
            # temp_calc = mod_data[mod_data>0]-np.min(mod_data[mod_data>0])
            # temp_calc = (temp_calc*(trained_model[mod_cnt].stdrange[-1]-trained_model[mod_cnt].stdrange[0])/np.max(temp_calc))+trained_model[mod_cnt].stdrange[0]
            mod_new_data = new_data[..., mod_cnt]
            mod_new_data[mod_data > 0] = trained_model.transform(mod_data[mod_data > 0],
                                                                 surpress_mapping_check=True)
            mod_new_data[mod_data == 0] = 0
        print("\rHL IRS setting", cur_cnt, end="")
    logthis("HL IRS setting finished!")


def get_mean_var(dataset, irs_dataset):
    logthis("Mean and variance started!")

    # irs_data = np.memmap(filename=HDD_OUTPUT_PATH + "%s_orig.dat" % (dataset), dtype=np.float32,
    #                      mode="r").reshape(-1, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3])
    irs_data = np.memmap(filename=HDD_OUTPUT_PATH + "%s_%s_irs.dat" % (dataset, irs_dataset), dtype=np.float32,
                         mode="r").reshape(-1, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3])
    irs_mean = np.zeros(shape=MOD_CNT, dtype=np.float64)

    for curdata in irs_data:
        irs_mean += np.sum(curdata, axis=(0, 1, 2))
    irs_mean /= np.prod(irs_data.shape[:-1])
    print(irs_mean)
    logthis("Mean finished!")

    irs_var = np.zeros(shape=MOD_CNT, dtype=np.float64)
    for data in irs_data:
        irs_var += np.sum(((data - irs_mean) ** 2), axis=(0, 1, 2))
    irs_var /= np.prod(irs_data.shape[:-1])
    print(irs_var)
    irs_var **= 0.5
    print(irs_var)
    logthis("Variance finished!")
    np.save(HDD_OUTPUT_PATH + "%s_%s_mv.npy" % (dataset, irs_dataset), np.array([irs_mean, irs_var]))


def get_patch_info(dataset):
    logthis("Patch info started!")
    alllabel = np.memmap(filename=HDD_OUTPUT_PATH + dataset + "_orig.lbl", dtype=np.uint8, mode="r").reshape(-1,
                                                                                                             SHAPE[0],
                                                                                                             SHAPE[1],
                                                                                                             SHAPE[2])

    data_size = alllabel.shape[0]

    lbl_bin = np.zeros(shape=(data_size, LABEL_CNT), dtype=np.uint32)
    for label, curcnt in zip(alllabel, range(data_size)):
        lbl_bin[curcnt] = np.bincount(label.flatten(), minlength=LABEL_CNT)
    np.save(HDD_OUTPUT_PATH + dataset + "_lblbin.npy", lbl_bin)


def get_extract_idx(dataset, extract_size):
    logthis("Index extraction started!")

    alldata = np.memmap(filename=HDD_OUTPUT_PATH + dataset + "_orig.dat", dtype=np.float32, mode="r").reshape(-1,
                                                                                                              SHAPE[0],
                                                                                                              SHAPE[1],
                                                                                                              SHAPE[2],
                                                                                                              SHAPE[3])
    alllabel = np.memmap(filename=HDD_OUTPUT_PATH + dataset + "_orig.lbl", dtype=np.uint8, mode="r",
                         shape=alldata.shape[:-1])

    data_size = alldata.shape[0]

    lbl_bin = np.load(HDD_OUTPUT_PATH + dataset + "_lblbin.npy")
    cut_size = (extract_size / data_size).astype(np.uint16)
    lbl_idx = np.zeros((data_size, LABEL_CNT), dtype=np.uint16)
    for curcnt in range(LABEL_CNT):
        temp_bin = np.minimum(cut_size[curcnt], lbl_bin[:, curcnt])
        allcnt = np.sum(temp_bin)
        while (allcnt <= extract_size[curcnt]):
            cut_size[curcnt] += 1
            temp_bin = np.minimum(cut_size[curcnt], lbl_bin[:, curcnt])
            allcnt = np.sum(temp_bin)
        lbl_idx[:, curcnt] = temp_bin
    result_idx = [np.empty(shape=(0, 4), dtype=np.uint16), np.empty(shape=(0, 4), dtype=np.uint16),
                  np.empty(shape=(0, 4), dtype=np.uint16), np.empty(shape=(0, 4), dtype=np.uint16),
                  np.empty(shape=(0, 4), dtype=np.uint16), ]
    for data, label, max_idx, curcnt in zip(alldata, alllabel, lbl_idx, range(data_size)):
        data = data.copy()
        label = label.copy()
        non_bg_idx = np.argwhere(data[..., 0] != 0).astype(np.uint8)
        max_bg = non_bg_idx.shape[0]
        for cur_mod_cnt in range(1, MOD_CNT):
            temp_bg_idx = np.argwhere(data[..., cur_mod_cnt] != 0).astype(np.uint8)
            if max_bg < temp_bg_idx.shape[0]:
                max_bg = temp_bg_idx.shape[0]
                non_bg_idx = temp_bg_idx
        cur_lbl_idx = np.argwhere(label == 0).astype(np.uint8)  # B

        cumdims = (np.maximum(non_bg_idx.max(), cur_lbl_idx.max()) + 1) ** np.arange(cur_lbl_idx.shape[1])
        cur_lbl_idx = non_bg_idx[np.in1d(non_bg_idx.dot(cumdims), cur_lbl_idx.dot(cumdims))].astype(np.uint8)
        cur_lbl_idx = np.sort(cur_lbl_idx.view("u1,u1,u1"), order=["f2", "f1", "f0"], axis=0).view(np.uint8)
        cur_lbl_idx = cur_lbl_idx[
            np.linspace(0, cur_lbl_idx.shape[0] - 1, max_idx[0], endpoint=True, retstep=False, dtype=np.uint32)]
        temp_idx = np.zeros(shape=(cur_lbl_idx.shape[0], 4), dtype=np.uint16)
        temp_idx[:, 0] = curcnt
        temp_idx[:, 1:] = cur_lbl_idx
        result_idx[0] = np.append(result_idx[0], temp_idx, axis=0)
        for curlbl in range(1, LABEL_CNT):
            cur_lbl_idx = np.argwhere(label == curlbl).astype(np.uint8).copy()
            if cur_lbl_idx.shape[0] == 0:
                continue
            cur_lbl_idx = np.sort(cur_lbl_idx.view("u1,u1,u1"), order=["f2", "f1", "f0"], axis=0).view(np.uint8)
            cur_lbl_idx = cur_lbl_idx[
                np.linspace(0, cur_lbl_idx.shape[0], max_idx[curlbl], endpoint=False, dtype=np.uint32)]
            temp_idx = np.zeros(shape=(cur_lbl_idx.shape[0], 4), dtype=np.uint16)
            temp_idx[:, 0] = curcnt
            temp_idx[:, 1:] = cur_lbl_idx
            result_idx[curlbl] = np.append(result_idx[curlbl], temp_idx, axis=0)
        print("\rIndex extraction %s:" % (dataset), curcnt + 1, "/", data_size,
              (result_idx[0].shape, result_idx[1].shape, result_idx[2].shape, result_idx[3].shape, result_idx[4].shape),
              end="")
    np.save(HDD_OUTPUT_PATH + dataset + "_lblidx.npz", result_idx)
    # result_idx = np.sort(result_idx.view("u1,u1,u1,u1"), order=["f0", "f1", "f2", "f3"], axis=0).astype(np.uint16)
    # np.save(HDD_OUTPUT_PATH + dataset + "_lblidx1.npy", result_idx[::2])
    # np.save(HDD_OUTPUT_PATH + dataset + "_lblidx2.npy", result_idx[1::2])
    #     result_idx = np.append(np.array([curcnt]), result_idx)
    #     print(result_idx.shape)
    #     result_idx[start_cnt:int(start_cnt + np.sum(max_idx)), 0] = curcnt
    #     result_idx[start_cnt:start_cnt+max_idx[0],1:] = cur_lbl_idx[np.linspace(0,cur_lbl_idx.shape[0],max_idx[0], endpoint=False,retstep=False, dtype=np.uint16)]
    #     start_cnt +=max_idx[0]
    #     for curlbl in range(1,LABEL_CNT):
    #         cur_lbl_idx = np.argwhere(label == curlbl).astype(np.uint8)
    #         result_idx[start_cnt:start_cnt+max_idx[curlbl], 1:] = cur_lbl_idx[np.linspace(0,cur_lbl_idx.shape[0],max_idx[curlbl], endpoint=False, dtype=np.uint16)]
    #         start_cnt+=max_idx[curlbl]
    #     print("\rIndex extraction %s:"%(dataset), curcnt+1,"/",data_size,end = "")
    #     result_idx = np.sort(result_idx.view("u1, u1,u1,u1"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint8)
    # np.save(HDD_OUTPUT_PATH+dataset+"_lblid1.npy", result_idx)
    # np.save(HDD_OUTPUT_PATH+dataset+"_lblidx1.npy", result_idx[::2])
    # np.save(HDD_OUTPUT_PATH+dataset+"_lblidx2.npy", result_idx[1::2])


def logthis(a):
    print("\n" + str(datetime.now()) + ": " + str(a))


if __name__ == "__main__":
    dataset = "hl"
    irs_dataset = "hl"
    extract_size = 4000000 * np.array([0.5, 0.125, 0.125, 0.125, 0.125])
    mean_size = 0
    patch_size = 33
    is_inception = True
###############################################################
# save_orig_data()
# train_IRS()
# set_IRS(irs_dataset=irs_dataset)
# get_mean_var(dataset=dataset, irs_dataset=irs_dataset)
# get_patch_info(dataset=dataset)
# get_extract_idx(dataset=dataset, extract_size=extract_size)
