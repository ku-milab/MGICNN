from __future__ import print_function

import os
import time
from datetime import datetime

#import medpy.io
import numpy as np
import tensorflow as tf


def init_weight_bias(name, shape, filtercnt, trainable):
    weights = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              dtype=tf.float32, trainable=trainable)
    biases = tf.Variable(initial_value=tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b",
                         trainable=trainable)
    return weights, biases


def conv_layer(data, weight, bias, padding, is_inception):
    conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
    if is_inception:
        return tf.nn.bias_add(conv, bias)
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


def conv3d_layer(data, weight, bias, padding, is_inception):
    conv = tf.nn.conv3d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
    if is_inception:
        return tf.nn.bias_add(conv, bias)
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


def pool_layer(data):
    return tf.nn.max_pool(value=data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


def fc_layer(data, weight, bias, dropout, batch_norm=False):
    shape = data.get_shape().as_list()
    shape = [shape[0], np.prod(shape[1:])]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
    if batch_norm:
        hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = tf.nn.relu(hidden)
    if dropout < 1.:
        hidden = tf.nn.dropout(hidden, dropout)
    return hidden


def output_layer(data, weight, bias, label):
    shape = data.get_shape().as_list()
    shape = [shape[0], np.prod(shape[1:])]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
    if label is None:
        return None, tf.nn.softmax(hidden)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(hidden, label)), tf.nn.softmax(hidden)


class model_def:
    def __init__(self):
        self.patch_size = 64
        self.mod_cnt = 1
        self.lbl_cnt = 2
        self.kernel_size = [5, 3]
        self.filters = [24, 32, 48, 16]

        self.batch_size = 128
        self.do_rate = 0.5

    def original_CNN(self, train=True):
        conv_layer_shape = [[self.kernel_size[0], self.kernel_size[0], self.mod_cnt, self.filters[0]],
                            [self.kernel_size[1], self.kernel_size[1], self.filters[0], self.filters[1]],
                            [self.kernel_size[1], self.kernel_size[1], self.filters[1], self.filters[2]]]

        fc_layer_shape = [[6 * 6 * 48, self.filters[3]], [self.filters[3], self.filters[3]], [self.filters[3], self.lbl_cnt]]

        if train:
            batch_size = self.batch_size
        else:
            batch_size = 128
        train_data_node = tf.placeholder(tf.float32, shape=(batch_size, self.patch_size, self.patch_size, self.mod_cnt))

        if train:
            do_rate = self.do_rate
            train_labels_node = tf.placeholder(tf.int64, shape=batch_size)
        else:
            do_rate = 1.
            train_labels_node = None

        cw = []
        cb = []

        fw = []
        fb = []

        layers = [train_data_node]

        cross_entropy, softmax = None, None
        for kernel, layer_cnt in zip(conv_layer_shape, range(len(conv_layer_shape))):
            w, b = init_weight_bias(name="c%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1], trainable=train)
            cw.append(w)
            cb.append(b)
        for kernel, layer_cnt in zip(fc_layer_shape, range(len(fc_layer_shape))):
            w, b = init_weight_bias(name="f%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1], trainable=train)
            fw.append(w)
            fb.append(b)
        for w, b, layer_cnt in zip(cw, cb, range(len(cw))):
            output = conv_layer(data=layers[-1], weight=w, bias=b, padding="VALID", is_inception=False)
            layers.append(output)
            if layer_cnt == 0 or layer_cnt == 1 or layer_cnt == 2:
                output = pool_layer(data=layers[-1])
                layers.append(output)
        for w, b, layer_cnt in zip(fw, fb, range(len(fw))):
            if layer_cnt == 2:
                cross_entropy, softmax = output_layer(data=layers[-1], weight=w, bias=b, label=train_labels_node)
            else:
                output = fc_layer(data=layers[-1], weight=w, bias=b, dropout=do_rate, batch_norm=False)
                layers.append(output)
        return cross_entropy, softmax, layers, train_data_node, train_labels_node


class model_execute:
    def __init__(self, train_dataset, val_dataset, test_dataset, dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.lists_data = dataset
        #self.input_shape = (-1, 240, 240, 155, 4)
        self.epochs = 200
        self.eval_freq = 100
        self.init_lr = 0.001
        self.final_lr = 0.000003

        self.patch_size = 64
        self.mod_cnt = 1
        self.lbl_cnt = 2
        self.batch_size = 128

        self.hdd_output_path = "output_test/"
        self.data_path = "output"

    def extract_patch_original_CNN(self):
        #logthis("Extract patch started!")
        logthis("DATA & Label ConCat!!")

        def find(target, obj):
            for i, lst in enumerate(obj):
                for j, name in enumerate(lst):
                    if name == target:
                        return i
            return (None, None)

        def count_data_size(target, lists):
            size = 1
            for i, name in enumerate(target):
                for list in lists:
                    if list[0] == name[5:-4]:
                        size += (int(list[1]) + 1)
                        print(list[1], size)
            return size

        if not os.path.exists(self.hdd_output_path):
            os.makedirs(self.hdd_output_path)

        def mk_patch_origial_CNN(dataset, lists_data, dataset_name, data_path):
            dataset_size = count_data_size(dataset, lists_data)
            pData = np.memmap(filename=self.hdd_output_path + dataset_name + "Total.dat",
                              dtype='float32', mode='w+', shape=(64, 64, dataset_size))
            plabel = np.memmap(filename=self.hdd_output_path + dataset_name +"Total.lbl",
                               dtype='uint8', mode='w+', shape=(1, 1, dataset_size))
            zStart = 0
            zEnd = 0
            for c, CTNum in enumerate(dataset):
                save_path = "%s/%s.dat" % (data_path, CTNum[5:-4])
                lbl_path = "%s/%s.lbl" % (data_path, CTNum[5:-4])
                data_size = find(CTNum[5:-4], lists_data)
                data = np.memmap(filename=data_path, dtype='float32', mode='r',
                                 shape=(64, 64, int(lists_data[data_size][1])))
                label = np.memmap(filename=lbl_path, dtype='uint8', mode='r',
                                  shape=(1, 1, int(lists_data[data_size][1])))

                if c == 0:
                    zStart = 0
                    zEnd = (zStart + int(lists_data[data_size][1]))
                else:
                    zStart = (zEnd + 1)
                    zEnd = (zStart + int(lists_data[data_size][1]))

                print(zStart, int(lists_data[data_size][1]), zEnd, data.shape)
                pData[:, :, zStart:zEnd] = data.copy()
                plabel[:, :, zStart:zEnd] = label.copy()

                del data, label
                print(CTNum)

            del pData, plabel

        mk_patch_origial_CNN(dataset=self.train_dataset, lists_data=self.lists_data, dataset_name="train",
                             data_path=self.data_path)
        mk_patch_origial_CNN(dataset=self.val_dataset, lists_data=self.lists_data, dataset_name="val",
                             data_path=self.data_path)
        mk_patch_origial_CNN(dataset=self.test_dataset, lists_data=self.lists_data, dataset_name="test",
                             data_path=self.data_path)



    def train_original_CNN(self, cross_entropy, softmax, data_node, label_node):
        logthis("Original CNN training started!")

        data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn" % (self.dataset, self.irs_dataset)
        lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
        data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
        data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)

        train_size = lbl.shape[0]

        batch = tf.Variable(0, dtype=tf.float32)  # LR*D^EPOCH=FLR --> LR/FLR
        learning_rate = tf.train.exponential_decay(learning_rate=self.init_lr, global_step=batch * self.batch_size,
                                                   decay_steps=train_size, staircase=True,
                                                   decay_rate=np.power(self.final_lr / self.init_lr,
                                                                       np.float(1) / self.epochs))
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)
        predict = tf.to_double(100) * (
            tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(softmax, label_node, 1))))

        allmean, allvar = np.load(self.hdd_output_path + "%s_%s_mv.npy" % (self.dataset, self.irs_dataset))
        with tf.Session() as sess:
            summary_path = self.hdd_output_path + "summary_%s_%s_orig_cnn/%d" % (
                self.dataset, self.irs_dataset, int(time.time()))
            model_path = self.hdd_output_path + "model_%s_%s_orig_cnn/" % (self.dataset, self.irs_dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            tf.initialize_all_variables().run()
            print("Variable Initialized")
            tf.scalar_summary("error", predict)
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(summary_path, sess.graph)
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=30)
            start_time = time.time()
            cur_epoch = 0
            cut_size = train_size - (train_size // self.batch_size) * self.batch_size
            randnum = np.random.randint(0, cut_size)
            curdata = data[randnum:train_size + randnum - cut_size]
            curlbl = lbl[randnum:train_size + randnum - cut_size]
            for step in range(int(self.epochs * train_size) // self.batch_size):
                # offset = (step * self.batch_size) % (train_size - self.batch_size)
                offset = step % self.batch_size
                batch_data = curdata[offset::curlbl.shape[0] / self.batch_size] * (allvar)
                batch_labels = curlbl[offset::curlbl.shape[0] / self.batch_size]
                feed_dict = {data_node: batch_data, label_node: batch_labels}

                _, l, lr, predictions, summary_out = sess.run(
                    [optimizer, cross_entropy, learning_rate, predict, summary_op],
                    feed_dict=feed_dict)

                summary_writer.add_summary(summary_out, global_step=step * self.batch_size)
                if step % self.eval_freq == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print(
                        'Step %d (epoch %.2f), %d s' % (step, float(step) * self.batch_size / train_size, elapsed_time))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % predictions)

                if cur_epoch != (
                            step * self.batch_size) / train_size:  # 1 epoch ((450000*4)/(128*100))*2min=281min=4.5hours.
                    print("Saved in path", saver.save(sess, model_path + "%d.ckpt" % (cur_epoch)))
                    randnum = np.random.randint(0, cut_size)
                    curdata = data[randnum:train_size + randnum - cut_size]
                    curlbl = lbl[randnum:train_size + randnum - cut_size]
                cur_epoch = (step * self.batch_size) / train_size
            print("Saved in path", saver.save(sess, model_path + "savedmodel_final.ckpt"))
        tf.reset_default_graph()

    def test_original_CNN(self, softmax, data_node, dataset, model_epoch=str(10)):
        logthis("Original CNN testing started!")

        # data_path = self.hdd_output_path + "%s_%s_irs.dat" % (dataset, self.irs_dataset)
        data_path = self.hdd_output_path + "%s_orig.dat" % (dataset)
        model_path = self.hdd_output_path + "model_%s_%s_orig_cnn/%s.ckpt" % (
            self.dataset, self.irs_dataset, model_epoch)
        pm_path = self.hdd_output_path + "%s_%s_pm_orig_cnn.dat" % (dataset, self.irs_dataset)
        img_path = self.hdd_output_path + "%s_%s_orig_cnn_output/" % (dataset, self.irs_dataset)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        data_num = get_filename(dataset)
        data = np.memmap(filename=data_path, dtype=np.float32, mode="r").reshape(self.input_shape)

        rolled_data = np.rollaxis(data, 0, -1)
        rolled_data = rolling_window(rolled_data, (self.patch_size, self.patch_size))
        rolled_data = np.rollaxis(rolled_data, 4, 7)
        rolled_data = np.rollaxis(rolled_data, 3)  # (110, 208, 208, 155, 33, 33, 4)
        allmean, allvar = np.load(self.hdd_output_path + "%s_%s_mv.npy" % (self.dataset, self.irs_dataset))

        pm_shape = rolled_data.shape[:4]
        result_shape = (data.shape[0], data.shape[1], data.shape[2], data.shape[3], self.mod_cnt + self.lbl_cnt)
        all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="w+", shape=result_shape)
        #img_hdr = medpy.io.load("/media/wltjr1007/hdd1/personal/brats/data/h.1.VSD.Brain.XX.O.MR_Flair.54512.nii")[1]
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            for curdata, curdata_cnt in zip(rolled_data, range(pm_shape[0])):
                result = np.zeros(shape=(pm_shape[1], pm_shape[2], pm_shape[3], self.lbl_cnt), dtype=np.float32)
                local_time = time.time()
                for curheight, curheight_cnt in zip(curdata, range(pm_shape[1])):
                    for curwidth, curwidth_cnt in zip(curheight, range(pm_shape[2])):
                        if np.all(curwidth == 0):
                            result[curheight_cnt, curwidth_cnt, ..., 0] = 1
                        feed_dict = {data_node: (curwidth - allmean) / allvar}
                        result[curheight_cnt, curwidth_cnt] = sess.run(softmax, feed_dict=feed_dict)
                    print("\rOriginal CNN testing", curdata_cnt + 1, "/", pm_shape[0], curheight_cnt, "/", pm_shape[1],
                          time.time() - local_time, end="")
                    local_time = time.time()
                result = np.pad(result, (
                    (self.patch_size // 2, self.patch_size // 2), (self.patch_size // 2, self.patch_size // 2), (0, 0),
                    (0, 0)), mode="edge")
                zero_idx = np.argwhere(data[curdata_cnt, ..., 0] == 0)
                result[zero_idx[:, 0], zero_idx[:, 1], zero_idx[:, 2], 0] = 1
                result[zero_idx[:, 0], zero_idx[:, 1], zero_idx[:, 2], 1:] = 0
               # medpy.io.save(np.argmax(result, axis=-1).astype(np.uint8), img_path + "VSD.%s_%s_%d_orig_cnn.%d.nii" % (
               #    dataset, self.irs_dataset, curdata_cnt, data_num[curdata_cnt]), img_hdr)
                all_result[curdata_cnt, ..., :self.mod_cnt] = (data[curdata_cnt] - allmean) / allvar
                all_result[curdata_cnt, ..., self.mod_cnt:] = result
        tf.reset_default_graph()

    def code_test(self):
        # data_path = self.hdd_output_path + "%s_orig.dat" % ("t")
        # data = np.memmap(filename=data_path, dtype=np.float32, mode="r").reshape(self.input_shape)

        #

        data_path = self.hdd_output_path + "%s_%s_irs.dat" % (self.dataset, self.irs_dataset)
        lbl_path = self.hdd_output_path + "%s_orig.lbl" % (self.dataset)
        data = np.memmap(filename=data_path, dtype=np.float32, mode="r").reshape(self.input_shape)
        label = np.memmap(filename=lbl_path, dtype=np.uint8, mode="r", shape=data.shape[:-1])

        # data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn2" % (self.dataset, self.irs_dataset)
        # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
        # data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
        # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
        lbl_idx_cls = np.load(self.hdd_output_path + "%s_lblidx.npz.npy" % (self.dataset))
        lbl_idx = np.empty(shape=(0, 4), dtype=np.uint16)
        for cur_cnt, curlbl in enumerate(lbl_idx_cls):
            templbl = np.sort(curlbl.view("u2,u2,u2,u2"), order=["f0", "f3", "f1", "f2"], axis=0).view(np.uint16)
            print(cur_cnt)
            if cur_cnt == 0:
                continue
            else:
                lbl_idx = np.append(lbl_idx, templbl[::2], axis=0)
        lbl_idx = np.sort(lbl_idx.view("u2,u2,u2,u2"), order=["f0", "f1", "f2", "f3"], axis=0).view(np.uint16)

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        ims = []
        fig = plt.figure(figsize=(24, 12))
        for a in range(10):
            templbl = lbl_idx[lbl_idx[:, 0] == a]
            for b in range(155):
                templbl2 = templbl[templbl[:, -1] == b]
                ax = plt.subplot(1, 1, 1)
                c = ax.imshow(data[a, :, :, b, 0], cmap="gray", animated=True)
                d = ax.scatter(templbl2[:, 2], templbl2[:, 1], facecolor="red", s=33 ** 2, marker="s",
                               c=label[templbl2[:, 0], templbl2[:, 1], templbl2[:, 2], templbl2[:, 3]])
                ims.append([c, d])

        #ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        #plt.show()
        #plt.close()
        #
        # import matplotlib.pyplot as plt
        # print(lbl[:128])
        # for a in range(128):
        #     ax = plt.subplot(8,16,a+1)
        #     ax.imshow(data[a,...,0], cmap="gray")
        #     ax.axis("off")
        # plt.show()

        # pm_path = self.hdd_output_path + "%s_%s_pm_orig_cnn.dat" % ("t", self.irs_dataset)
        # result_shape = (self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3],self.mod_cnt+self.lbl_cnt)
        # all_result = np.memmap(filename=pm_path, dtype=np.float32, mode="r").reshape(result_shape)
        # import matplotlib.pyplot as plt
        # for a in range(9):
        #     if a<4:
        #         ax = plt.subplot(2,4,a+1)
        #         ax.imshow(all_result[0,:,:,70,a], cmap="gray")
        #     else:
        #         ax = plt.subplot(2, 5, a + 2)
        #         ax.imshow(all_result[0,:,:,70,a], vmin=0, vmax=1)
        # plt.show()

        # data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn" % (self.dataset, self.irs_dataset)
        # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
        # data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
        # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
        #
        # import matplotlib.pyplot as plt
        # import matplotlib.animation as animation
        #
        # for i in range(0, data_shape[0],128):
        #     ims = []
        #     fig = plt.figure(figsize=(24,12))
        #     for j in range(4):
        #         im = []
        #         for curidx, curdata in enumerate(data[i:i+128]):
        #             ax = plt.subplot(8,16,curidx+1)
        #             a = ax.imshow(curdata[...,j], cmap="gray")
        #             ax.axis("off")
        #             im.append(a)
        #         ims.append(im)
        #
        #     ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
        #     plt.show()
        #     plt.close()




        # data_path = self.hdd_output_path + "%s_orig" % (self.dataset)
        # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r").reshape(self.input_shape)
        # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r", shape=data.shape[:-1])
        #
        # lbl_idx_cls = np.load(self.hdd_output_path + "%s_lblidx.npz.npy" % (self.dataset))
        # lbl_idx = np.empty(shape=(0,4), dtype=np.uint16)
        # for curlbl in lbl_idx_cls:
        #     templbl = np.sort(curlbl.view("u2,u2,u2,u2"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint16)
        #     lbl_idx = np.append(lbl_idx, templbl[::2], axis=0)
        # # lbl_idx = np.sort(lbl_idx.view("u2,u2,u2,u2"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint16)
        # lbl_idx = np.sort(lbl_idx.view("u2,u2,u2,u2"), order=["f0", "f1", "f2", "f3"], axis=0).view(np.uint16)
        #
        #
        # import matplotlib.pyplot as plt
        # import matplotlib.animation as animation
        #
        # ims = []
        # fig = plt.figure(figsize=(24, 12))
        # for curcnt in range(0,274,5):
        #     cur_idx = lbl_idx[lbl_idx[:,0]==curcnt]
        #     for curheight in range(155):
        #         cur_idx_height = cur_idx[cur_idx[:,-1]==curheight][:,1:3]
        #         ax1 = plt.subplot(2, 3, 1)
        #         ax2 = plt.subplot(2, 3, 2)
        #         ax3 = plt.subplot(2, 3, 3)
        #         ax4 = plt.subplot(2, 3, 4)
        #         ax5 = plt.subplot(2, 3, 5)
        #
        #         a=ax1.imshow(data[curcnt,...,curheight,0], cmap="gray", animated=True)
        #         b=ax2.imshow(data[curcnt,...,curheight,1], cmap="gray", animated=True)
        #         c=ax3.imshow(data[curcnt,...,curheight,2], cmap="gray", animated=True)
        #         d=ax4.imshow(data[curcnt,...,curheight,3], cmap="gray", animated=True)
        #         e = ax5.imshow(lbl[curcnt,...,curheight], vmin=0, vmax=4)
        #         a1 = ax1.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
        #         a2 = ax2.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
        #         a3 = ax3.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
        #         a4 = ax4.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
        #         a5 = ax5.scatter(cur_idx_height[:,1],cur_idx_height[:,0], facecolor="red")
        #         ims.append([a,b,c,d,e,a1,a2,a3,a4,a5])
        #     print(curcnt)
        #
        # ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
        # plt.show()
        # plt.close()


        # data_path2 = self.hdd_output_path + "%s_orig" % (self.dataset)
        # data2 = np.memmap(filename=data_path2 + ".dat", dtype=np.float32, mode="r").reshape(self.input_shape)
        #
        # allmean, allvar = np.load(self.hdd_output_path + "%s_%s_mv.npy" % (self.dataset, self.irs_dataset))
        # print(allmean, allvar)
        # for dat in data2:
        #     print(np.mean((dat-allmean)/allvar,axis=(0,1,2)),np.var((dat-allmean)/allvar,axis=(0,1,2)))

        # data_path = self.hdd_output_path + "%s_%s_patch_orig_cnn" % (self.dataset, self.irs_dataset)
        # lbl = np.memmap(filename=data_path + ".lbl", dtype=np.uint8, mode="r")
        # data_shape = (lbl.shape[0], self.patch_size, self.patch_size, self.mod_cnt)
        # data = np.memmap(filename=data_path + ".dat", dtype=np.float32, mode="r", shape=data_shape)
        #
        # allmean, allvar = np.load(self.hdd_output_path + "%s_%s_mv.npy" % (self.dataset, self.irs_dataset))
        # cnt = 0
        # print(np.mean(data,axis=(0,1,2)))
        # print(np.var(data[:data.shape[0]//2], axis=(0,1,2)))
        # for dat, curcnt in zip(data, range(data.shape[0])):
        #     for mod in range(self.mod_cnt):
        #         print(curcnt, mod, np.mean(dat[...,mod]), np.var(dat[...,mod]))
        # import matplotlib.pyplot as plt
        #
        # import matplotlib.animation as animation
        # # ims = []
        # # fig = plt.figure(figsize=(24,12))
        # print(data.shape)
        # for dat in data:
        #     fig = plt.figure(figsize=(24,12))
        #     ims=[]
        #     for curdepth in range(155):
        #         ax1 = plt.subplot(1,4,1)
        #         ax2 = plt.subplot(1,4,2)
        #         ax3 = plt.subplot(1,4,3)
        #         ax4 = plt.subplot(1,4,4)
        #         a =ax1.imshow(dat[...,curdepth, 0], cmap="gray", animated=True)
        #         b= ax2.imshow(dat[...,curdepth, 1], cmap="gray", animated=True)
        #         c= ax3.imshow(dat[...,curdepth, 2], cmap="gray", animated=True)
        #         d= ax4.imshow(dat[...,curdepth, 3], cmap="gray", animated=True)
        #         ims.append([a,b,c,d])
        #
        #     ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
        #     plt.show()
        #     plt.close()

        # currently 13240 patches are blank. WHY????
        # 274 images, max idx is 255
        # lbl_idx = np.load(self.hdd_output_path + "%s_lblidx2.npy" % (self.dataset))
        # lbl_idx = np.sort(lbl_idx.view("u1,u1,u1,u1"), order=["f3", "f2", "f1", "f0"], axis=0).view(np.uint8)
        # for a in np.argmax(lbl_idx, axis=0):
        #     print(lbl_idx[a])


def rolling_window_lastaxis(a, window):
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a


def get_filename(dataset):
    # h.86.VSD.Brain.XX.O.MR_T1.35779.nii
    name_list = os.listdir("../../../../data/")
    result = {}
    for name in name_list:
        temp = name.replace(".mhd", "").split(".")
        if temp[0] != dataset or temp[-2] != "MR_Flair":
            continue
        result[int(temp[1]) - 1] = int(temp[-1])
    return result


def logthis(a):
    print("\n" + str(datetime.now()) + ": " + str(a))
