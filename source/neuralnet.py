import os, acoustics
import numpy as np
import tensorflow as tf
import source.layers as lay

class SeqAE(object):

    def __init__(self, \
        seq_len, seq_dim, zdim, \
        learning_rate=1e-3, path='', verbose=True):

        print("\nInitializing Neural Network...")
        self.seq_len, self.seq_dim, self.zdim = seq_len, seq_dim, zdim
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.seq_len, self.seq_dim], \
            name="x")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="batch_size")
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[], \
            name="training")

        self.layer = lay.Layers()

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

        feed_tr = {self.x:x, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.batch_size:x.shape[0], self.training:False}

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
            self.sess.run([self.variables['y_hat'], self.losses['loss']], \
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
                    self.loss_l2(self.variables['y_hat'] - self.x, [2])
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
            self.__encoder(x=self.x, outdim=self.zdim, reuse=False, \
            name='enc', verbose=verbose)

        noise = acoustics.generator.noise(N=self.zdim, \
            color='white', state=None)

        self.variables['c_j'] = self.variables['c_a'] #+ noise

        self.variables['y_hat'] = \
            self.__decoder(z=self.variables['c_a'], reuse=False, \
            name='dec', verbose=verbose)

    def __encoder(self, x, outdim=1, reuse=False, \
        name='enc', depth=2, verbose=True):

        if(verbose): print("\n* Encoder: %s" %(name))
        e = None
        with tf.compat.v1.variable_scope(name, reuse=reuse):

            x = tf.compat.v1.transpose(x, perm=[1, 0, 2], name="x_seq")
            h_now_1, c_now_1 = None, None
            h_now_2, c_now_2 = None, None

            for seq_idx in range(self.seq_len):
                if(seq_idx != 0): verbose = False

                h_now_1, c_now_1, y_now_1 = self.lstm_cell(x_now=x[seq_idx, :, :], \
                    h_prev=h_now_1, c_prev=c_now_1, output_len=self.zdim, \
                    training=self.training, activation="tanh", \
                    name="%s_lstm-1" %(name), verbose=verbose)
                h_now_2, c_now_2, y_now_2 = self.lstm_cell(x_now=y_now_1, \
                    h_prev=h_now_2, c_prev=c_now_2, output_len=self.zdim, \
                    training=self.training, activation="tanh", \
                    name="%s_lstm-2" %(name), verbose=verbose)

                z = tf.compat.v1.expand_dims(y_now_2, 0)
                if(e is None): e = z
                else: e = tf.concat([e, z], 0)

            return e

    def __decoder(self, z, reuse=False, \
        name='dec', depth=2, verbose=True):

        if(verbose): print("\n* Decoder: %s" %(name))
        d = None
        with tf.compat.v1.variable_scope(name, reuse=reuse):

            h_now_1, c_now_1 = None, None
            h_now_2, c_now_2 = None, None

            for seq_idx in range(self.seq_len):
                if(seq_idx != 0): verbose = False

                h_now_1, c_now_1, y_now_1 = self.lstm_cell(x_now=z[seq_idx, :, :], \
                    h_prev=h_now_1, c_prev=z[-1, :, :], output_len=self.seq_dim, \
                    training=self.training, activation="tanh", \
                    name="%s_lstm-1" %(name), verbose=verbose)
                h_now_2, c_now_2, y_now_2 = self.lstm_cell(x_now=y_now_1, \
                    h_prev=h_now_2, c_prev=c_now_2, output_len=self.seq_dim, \
                    training=self.training, activation="sigmoid", \
                    name="%s_lstm-2" %(name), verbose=verbose)

                y_hat = tf.compat.v1.expand_dims(y_now_2, 0)
                if(d is None): d = y_hat
                else: d = tf.concat([d, y_hat], 0)

            d = tf.compat.v1.transpose(d, perm=[1, 0, 2], name="y_hat")
            return d

    def lstm_cell(self, x_now, h_prev, c_prev, output_len, \
        training, activation="tanh", name="", verbose=True):

        if(h_prev is None): h_prev = tf.zeros_like(x_now)
        if(c_prev is None): c_prev = tf.zeros_like(x_now)

        i_term1 = self.layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
            training=self.training, activation="None", name="%s-i-term1" %(name), verbose=verbose)
        i_term2 = self.layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
            training=self.training, activation="None", name="%s-i-term2" %(name), verbose=verbose)
        i_now = self.layer.fully_connected(x=i_term1 + i_term2, c_out=x_now.shape[-1], \
            training=self.training, activation="sigmoid", name="%s-i" %(name), verbose=verbose)

        f_term1 = self.layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
            training=self.training, activation="None", name="%s-f-term1" %(name), verbose=verbose)
        f_term2 = self.layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
            training=self.training, activation="None", name="%s-f-term2" %(name), verbose=verbose)
        f_now = self.layer.fully_connected(x=f_term1 + f_term2, c_out=x_now.shape[-1], \
            training=self.training, activation="sigmoid", name="%s-f" %(name), verbose=verbose)

        o_term1 = self.layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
            training=self.training, activation="None", name="%s-o-term1" %(name), verbose=verbose)
        o_term2 = self.layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
            training=self.training, activation="None", name="%s-o-term2" %(name), verbose=verbose)
        o_now = self.layer.fully_connected(x=o_term1 + o_term2, c_out=x_now.shape[-1], \
            training=self.training, activation="sigmoid", name="%s-o" %(name), verbose=verbose)

        c_term1_1 = self.layer.fully_connected(x=x_now, c_out=x_now.shape[-1], \
            training=self.training, activation="None", name="%s-c-term1_1" %(name), verbose=verbose)
        c_term1_2 = self.layer.fully_connected(x=h_prev, c_out=h_prev.shape[-1], \
            training=self.training, activation="None", name="%s-c-term2_1" %(name), verbose=verbose)
        c_term1_sum = self.layer.activation(x=c_term1_1 + c_term1_2, \
            activation="tanh", name="%s-c-term1-sum" %(name))

        c_term1 = tf.compat.v1.multiply(i_now, c_term1_sum, name="%s-c-term1" %(name))
        c_term2 = tf.compat.v1.multiply(f_now, c_prev, name="%s-c-term2" %(name))
        c_now = tf.compat.v1.add(c_term1, c_term2, name="%s-c" %(name))

        h_now = tf.compat.v1.multiply(o_now, \
            self.layer.activation(x=c_now, activation="tanh", name="%s-c-act" %(name)), \
            name="%s-h" %(name))

        y_now = self.layer.fully_connected(x=h_now, c_out=output_len, \
            training=self.training, activation=activation, name="%s-y" %(name), verbose=verbose)

        return h_now, c_now, y_now
