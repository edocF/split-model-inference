import os

import torch
import torch.nn as nn
import torch.utils.data as data

from data_processing.data_pipeline import collate_fn_valid, GreedyDecoder
from data_processing.utils import cer, wer
from distributed_setup.client import *
from distributed_setup.server import *
from encoder.base_encoder import EncoderDecoder
from train_and_test.test_head import run_node0
from train_and_test.test_tail import run_node1


def test(model, sp_model, device, test_loader, criterion, encoder_decoder: EncoderDecoder, rank, host, port):
    print('\nevaluating...')

    # Non-split local evaluation path
    if sp_model is None and model is not None:
        model.eval()
        test_loss = 0
        test_cer, test_wer = [], []
        with torch.no_grad():
            for batch_idx, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                output = model(spectrograms)
                output = nn.functional.log_softmax(output, dim=2)
                output = output.transpose(0, 1)
                loss = criterion(output, labels, input_lengths, label_lengths)
                decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                test_loss += loss.item()
        avg_loss = test_loss / max(1, len(test_loader))
        avg_cer = sum(test_cer) / len(test_cer) if test_cer else 0.0
        avg_wer = sum(test_wer) / len(test_wer) if test_wer else 0.0
        print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(avg_loss, avg_cer, avg_wer))
        return

    # Split evaluation path
    if rank == 0:
        s = set_client_connection(host, port)
        run_node0(model, sp_model, device, test_loader, encoder_decoder, s)
        s.close()
    elif rank == 1:
        server_socket = set_server_connection(host, port)
        run_node1(len(test_loader), model, sp_model, encoder_decoder, criterion, server_socket)


def evaluate_model(hparams, model, sp_model, test_dataset, encoder_decoder: EncoderDecoder, rank, host, port):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 0, 'pin_memory': False}
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=collate_fn_valid,
                                  **kwargs)
    criterion = nn.CTCLoss(blank=28).to(device)

    test(model, sp_model, device, test_loader, criterion, encoder_decoder, rank, host, port)
