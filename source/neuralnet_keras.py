import os, acoustics
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import LSTM, RepeatVector

class SeqAE(object):

    def __init__(self, \
        seq_len, seq_dim, zdim, \
        learning_rate=1e-3, path='', verbose=True):

        print("\nInitializing Neural Network...")
        self.seq_len, self.seq_dim, self.zdim = seq_len, seq_dim, zdim
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, None, self.seq_dim], \
            name="x")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="batch_size")
        self.sequence_len = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="sequence_len")

        self.variables, self.losses = {}, {}
        self.__build_model(verbose=verbose)
        self.__build_loss()

        with tf.control_dependencies(self.variables['ops']):
            self.optimizer = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, name='Adam').minimize(\
                self.losses['loss'], var_list=self.variables['params'])

        tf.compat.v1.summary.scalar('SeqAE/loss', self.losses['loss'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.__init_session(path=self.path_ckpt)

    def step(self, x, iteration=0, training=False):

        feed_tr = {self.x:x, self.batch_size:x.shape[0], self.sequence_len:x.shape[1]}
        feed_te = {self.x:x, self.batch_size:x.shape[0], self.sequence_len:x.shape[1]}

        summary_list = []
        if(training):
            try:
                _, summaries = self.sess.run([self.optimizer, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)
            except:
                _, summaries = self.sess.run([self.optimizer, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)

            for summaries in summary_list:
                self.summary_writer.add_summary(summaries, iteration)

        y_hat, loss = \
            self.sess.run([self.y_hat, self.losses['loss']], \
            feed_dict=feed_te)

        outputs = {'y_hat':y_hat, 'loss':loss}
        return outputs

    def save_parameter(self, model='model_checker', epoch=-1):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        if(epoch >= 0): self.summary_writer.add_run_metadata(self.run_metadata, 'epoch-%d' % epoch)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, verbose=True):

        print("\n* Parameter arrange")

        ftxt = open("list_parameters.txt", "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def loss_l2(self, x, reduce=None):

        distance = tf.compat.v1.reduce_sum(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-9), axis=reduce)

        return distance

    def __init_session(self, path):

        try:
            print("\n* Initializing Session")
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
        except: pass

    def __build_loss(self):

        self.losses['loss'] = \
            tf.compat.v1.reduce_mean(
                tf.compat.v1.reduce_sum(
                    self.loss_l2(self.variables['y_hat'] - self.x[:, ::-1, :], [2])
                , [1])
            )

        self.variables['params'] = []
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            self.variables['params'].append(var)

        self.variables['ops'] = []
        for ops in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS):
            self.variables['ops'].append(ops)

    def __build_model(self, verbose=True):

        self.variables['c_a'] = \
            self.__encoder(x=self.x, name='enc', verbose=verbose)

        noise = acoustics.generator.noise(N=self.zdim, \
            color='white', state=None)

        self.variables['c_j'] = self.variables['c_a'] #+ noise

        self.variables['y_hat'] = \
            self.__decoder(z=self.variables['c_j'], name='dec', verbose=verbose)
        self.y_hat = tf.add(self.variables['y_hat'][:, ::-1, :], 0, name="y_hat")

    def __encoder(self, x, name='enc', depth=2, verbose=True):

        if(verbose): print("\n* Encoder: %s" %(name))
        e = []

        self.list_dim = []
        for idx_d in range(depth):
            outdim = int(self.zdim*(0.5**(depth-idx_d-1)))
            self.list_dim.append(outdim)
            lstm = LSTM(units=self.list_dim[-1], kernel_initializer='he_normal',\
                return_sequences=True, return_state=True, name='%s_lstm-%d' %(name, idx_d))
            repeat = RepeatVector(self.sequence_len)

            if(verbose): print(x.shape, 'to', outdim)
            x, h, c = lstm(x)
            e = repeat(c)

        return e

    def __decoder(self, z, name='dec', depth=2, verbose=True):

        if(verbose): print("\n* Decoder: %s" %(name))

        for idx_d in range(depth):
            if(idx_d == depth-1): outdim = self.seq_dim
            else: outdim = self.list_dim[-(1+idx_d)]
            lstm = LSTM(units=outdim, kernel_initializer='he_normal', \
                return_sequences=True, return_state=False, name='%s_lstm-%d' %(name, idx_d))

            if(verbose): print(z.shape, 'to', outdim)
            z = lstm(z)

        d = tf.compat.v1.nn.sigmoid(z)
        # d = tf.clip_by_value(z, clip_value_min=0, clip_value_max=1)
        return d
