"""
Test the ESIM model on some preprocessed dataset.
"""
# Aurelien Coet, 2018.

import os
import json
import time
import pickle
import argparse
import torch
import numpy as np
from mfae.data import ElmoDataset
from torch.utils.data import DataLoader
from mfae.model_elmo2 import ESIM
from mfae.utils import correct_predictions
from sklearn import metrics
from allennlp.modules.elmo import batch_to_ids


def transform_elmo_data(data, batch_size=128, shuffle=True):
    data_batch = dict()
    data_batch['premises'] = dict()
    data_batch['hypotheses'] = dict()
    data_batch['labels'] = dict()
    index = np.arange(len(data['labels']))
    if shuffle:
        np.random.shuffle(index)

    idx = -1
    for i in range(len(index)):
        if i % batch_size == 0:
            idx += 1
            data_batch['premises'][idx] = []
            data_batch['hypotheses'][idx] = []
            data_batch['labels'][idx] = []
        data_batch['premises'][idx].append(data['premises'][index[i]])
        data_batch['hypotheses'][idx].append(data['hypotheses'][index[i]])
        data_batch['labels'][idx].append(int(data['labels'][index[i]]))
    for i in range(len(data_batch['labels'])):
        data_batch['premises'][i] = batch_to_ids(data_batch['premises'][i])
        data_batch['hypotheses'][i] = batch_to_ids(data_batch['hypotheses'][i])
    return data_batch

def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    all_labels = []
    all_out_classes = []

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises_ids = batch["premises"].squeeze().to(device)
            hypotheses_ids = batch["hypotheses"].squeeze().to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            all_labels.extend(labels.tolist())
            labels = labels.to(device)

            _, probs = model(premises_ids, hypotheses_ids)
            _, out_classes = probs.max(dim=1)

            all_out_classes.extend(out_classes.tolist())

            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy_score = metrics.accuracy_score(all_labels, all_out_classes)
    # precision_score = metrics.precision_score(all_labels, all_out_classes)
    # recall_score = metrics.recall_score(all_labels, all_out_classes)
    # f1_score = metrics.f1_score(all_labels, all_out_classes)

    return batch_time, total_time, accuracy_score#, precision_score, recall_score, f1_score


def main(test_file,
         options_file,
         weight_file,
         embedding_size=512,
         hidden_size=512,
         dropout=0.5,
         num_classes=3,
         batch_size=32,
         checkpoint=None):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    print("\t* Loading validation data...")
    with open(test_file, "rb") as pkl:
        test_data = pickle.load(pkl)
        test_data = transform_elmo_data(test_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(ElmoDataset(test_data), batch_size=1, shuffle=False)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    # with open(embeddings_file, "rb") as pkl:
    #     embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)

    model = ESIM(embedding_size,
                 hidden_size,
                 options_file, weight_file,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        model.load_state_dict(checkpoint["model"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy_score = test(
        model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy_score: {:.4f}%".format(batch_time, total_time, (accuracy_score*100)))


if __name__ == "__main__":

    default_config = "../../config/training/snli_training_elmo.json"

    parser = argparse.ArgumentParser(
        description="Train the ESIM model on quora")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = script_dir + '/scripts/training'

    parser.add_argument("--checkpoint",
                        default=os.path.dirname(os.path.realpath(__file__)) + '/data/checkpoints/SNLI/elmo/' +"best.pth.tar",
                        help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)


    main(os.path.normpath(os.path.join(script_dir, config["test_data"])),
         os.path.normpath(os.path.join(script_dir, config["options_file"])),
         os.path.normpath(os.path.join(script_dir, config["weight_file"])),
         config["embedding_size"],
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         32,
         args.checkpoint)
