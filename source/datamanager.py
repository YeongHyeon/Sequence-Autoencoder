import os, glob
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, datapath='dataset_npz'):

        print("\nInitializing Dataset...")

        self.list_npz = self.sorted_list(os.path.join(datapath, '*.npz'))
        self.list_npz = shuffle(self.list_npz)

        bound = int(len(self.list_npz) * 0.8)
        self.list_tr, self.list_te = self.list_npz[:bound], self.list_npz[bound:]
        self.num_tr, self.num_te = len(self.list_tr), len(self.list_te)

        print("\nDataset")
        print(" Training : %5d" %(self.num_tr))
        print(" Test     : %5d" %(self.num_te))

        self.reset_idx()
        x, _, _ = self.next_batch(batch_size=1, train=True)
        self.reset_idx()

        self.seq_len, self.seq_dim = x.shape[1], x.shape[2]
        print("\nSequence")
        print(" Length    : %3d" %(self.seq_len))
        print(" Dimension : %3d" %(self.seq_dim))

    def sorted_list(self, path):

        tmplist = glob.glob(path)
        tmplist.sort()

        return tmplist

    def reset_idx(self):

        self.list_tr = shuffle(self.list_tr)
        self.idx_tr, self.idx_te = 0, 0

    def load_npz(self, path):

        return np.load(path)

    def next_batch(self, batch_size=1, train=False):

        if(train): idx_d, num_d, list_d = self.idx_tr, self.num_tr, self.list_tr
        else: idx_d, num_d, list_d = self.idx_te, self.num_te, self.list_te

        batch_x, batch_y, terminate = [], [], False
        while(True):
            try:
                npz = self.load_npz(path=list_d[idx_d])
            except:
                if(train): self.list_tr = shuffle(self.list_tr)
                idx_d, terminate = 0, True
                break
            else:
                batch_x.append(np.asarray([npz['coord_x'], npz['coord_y']]).T)
                batch_y.append(npz['label'])
                idx_d += 1
                if(len(batch_x) >= batch_size): break

        if(train): self.idx_tr = idx_d
        else: self.idx_te = idx_d

        return np.asarray(batch_x), batch_y, terminate
