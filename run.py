import os, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import source.datamanager as dman
import source.neuralnet_keras as nn
import source.tf_process as tfp

def main():

    dataset = dman.Dataset(datapath=FLAGS.datapath)

    neuralnet = nn.SeqAE(seq_len=dataset.seq_len, seq_dim=dataset.seq_dim, zdim=FLAGS.zdim, learning_rate=FLAGS.lr, path='Checkpoint')

    neuralnet.confirm_params(verbose=False)

    # tfp.training(neuralnet=neuralnet, dataset=dataset, \
    #     epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    tfp.test(neuralnet=neuralnet, dataset=dataset, \
        batch_size=FLAGS.batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='dataset_npz', help='Dataset path')
    parser.add_argument('--zdim', type=int, default=128, help='Dimension of latent vector z')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch', type=int, default=1, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()
