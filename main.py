import argparse
from pathlib import Path

import torchaudio
try:
    # Prefer soundfile backend on Windows to read FLAC files
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

from encoder.AutoEncoders import AutoEncoders
from encoder.huffman import Huffman
from model.save_load_model import *
from train_and_test.test import evaluate_model
from train_and_test.train import train_model
from train_and_test.train_autoencoders import train_autoencoders


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_encoder(encoder_type, encoder_path, hparams):
    if encoder_type == 'huffman':
        print('Huffman Encoder is being used!')
        return Huffman()
    elif encoder_type == 'autoencoder':
        print('AutoEncoder is being used!')
        return AutoEncoders(
            encoder_path,
            input_size=hparams['input_layers'],
            hidden_size=hparams['hidden_layers'],
            output_size=hparams['output_layers'],
            leaky_relu=hparams['leaky_relu']
        )
    else:
        return None


def create_folder(path):
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        directory.mkdir(parents=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for split model inference')
    parser.add_argument('-split_mode', nargs='?', const=True, default=False, type=str2bool, required=False,
                        help='Enable split inference. Use "-split_mode" for True or "-split_mode False"')
    parser.add_argument('-host', metavar='host', action='store',
                        default="node0.grp19-cs744-3.uwmadison744-f20-pg0.wisc.cloudlab.us", required=False,
                        help='Hostname to connect')
    parser.add_argument('-port', metavar='Batch Size', action='store', default=60009, required=False,
                        help='Port to be used')
    parser.add_argument('-test', action='store_true', default=False, required=False, help='Test mode')
    parser.add_argument('-path', metavar='base-path', action='store', default="./", required=False,
                        help='The base path for the project')
    parser.add_argument('-batch', metavar='Batch Size', action='store', default=10, required=False,
                        help='Batch size to be used in training set')
    parser.add_argument('-epochs', metavar='Epochs', action='store', default=10, required=False,
                        help='No of Epochs for training')
    parser.add_argument('-savefile', metavar='Save File', action='store', default='model.pth', required=False,
                        help='File for saving the checkpoint')
    parser.add_argument('-encoder', metavar='Encoder type', action='store', default='huffman', required=False,
                        help='Encoder to be used encoding in split model inference')
    parser.add_argument('-encoderpath', metavar='Path of saved autoencoder model', action='store',
                        default='autoencoder.pth', required=False, help='Path of the saved models of autoencoder and '
                                                                        'decoder')
    parser.add_argument('-rank', metavar='Rank of node', action='store', default=0, required=False,
                        help='Rank of the node')
    # model size hyperparameters
    parser.add_argument('-n_cnn_layers', metavar='CNN layers', action='store', type=int, default=3, required=False,
                        help='Number of residual CNN layers')
    parser.add_argument('-n_rnn_layers', metavar='RNN layers', action='store', type=int, default=3, required=False,
                        help='Number of BiGRU layers')
    parser.add_argument('-rnn_dim', metavar='RNN dim', action='store', type=int, default=256, required=False,
                        help='Hidden size for GRU and FC output from CNN stack')
    parser.add_argument('-dropout', metavar='Dropout', action='store', type=float, default=0.1, required=False,
                        help='Dropout ratio')
    # autoencoder sizes (used when -encoder autoencoder)
    parser.add_argument('-ae_hidden', metavar='AE hidden dim', action='store', type=int, default=None, required=False,
                        help='Autoencoder hidden dim (default: rnn_dim//4)')
    parser.add_argument('-ae_output', metavar='AE bottleneck dim', action='store', type=int, default=None, required=False,
                        help='Autoencoder bottleneck dim (default: rnn_dim//16)')
    args = parser.parse_args()

    port = int(args.port)
    host = args.host

    # derive autoencoder dims from rnn_dim if not explicitly set
    derived_ae_hidden = args.ae_hidden if args.ae_hidden is not None else max(32, int(int(args.rnn_dim) // 4))
    derived_ae_output = args.ae_output if args.ae_output is not None else max(8, int(int(args.rnn_dim) // 16))

    hparams = {
        "n_cnn_layers": int(args.n_cnn_layers),
        "n_rnn_layers": int(args.n_rnn_layers),
        "rnn_dim": int(args.rnn_dim),
        "n_class": 29,
        "n_feats": 64,
        "stride": 2,
        "dropout": float(args.dropout),
        "learning_rate": 5e-4,
        "batch_size": int(args.batch),
        "epochs": int(args.epochs),
        # AE tied to rnn_dim by default
        "input_layers": int(args.rnn_dim),
        "hidden_layers": int(derived_ae_hidden),
        "output_layers": int(derived_ae_output),
        "leaky_relu": 0.2
    }
    node_rank = int(args.rank)
    if node_rank < 0 or node_rank > 1:
        raise Exception('Rank is incorrect. It should be either 0 or 1!')

    base_dataset_directory = "{}/dataset".format(args.path)
    create_folder(base_dataset_directory)
    train_dataset = None
    if not args.test:
        train_dataset = torchaudio.datasets.LIBRISPEECH(base_dataset_directory, url='train-clean-100',
                                                        download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(base_dataset_directory, url='test-clean', download=True)

    save_filepath = '{}/{}'.format(args.path, args.savefile)
    encoder_base_path = '{}/{}'.format(args.path, args.encoderpath)
    if args.test:
        encoder = get_encoder(args.encoder, encoder_base_path, hparams)
        if not bool(args.split_mode):
            print('Evaluating complete model without any splitting')
            model = load_model(save_filepath, hparams)
            evaluate_model(hparams, model, None, test_dataset, encoder, node_rank, host, port)
        else:
            print('Evaluating split model')
            sp_model = load_split_model(save_filepath, hparams)
            evaluate_model(hparams, None, sp_model, test_dataset, encoder, node_rank, host, port)
    else:
        if args.encoder == 'autoencoder':
            sp_model = load_split_model(save_filepath, hparams)
            model = train_autoencoders(sp_model, hparams, train_dataset, save_filepath=encoder_base_path)
        else:
            model = train_model(hparams, train_dataset, test_dataset, save_filepath=save_filepath)
