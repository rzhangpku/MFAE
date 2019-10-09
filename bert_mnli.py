"""
Train the ESIM model on the preprocessed SNLI dataset.
"""
# Aurelien Coet, 2018.

from utils_bert import train, validate
from mfae.model_bert import ESIM
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json
import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use('Agg')


def transform_batch_data(data, batch_size=64, shuffle=True):
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
    return data_batch

def main(train_file,
         valid_matched_file,
         valid_mismatched_file,
         target_dir,
         embedding_size=512,
         hidden_size=512,
         dropout=0.5,
         num_classes=3,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None):
    """
    Train the ESIM model on the Quora dataset.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the model.
        valid_file: A path to some preprocessed data that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = pickle.load(pkl)

    print("\t* Loading validation data...")
    with open(valid_matched_file, "rb") as pkl:
        valid_matched_data = pickle.load(pkl)
        valid_matched_dataloader = transform_batch_data(valid_matched_data, batch_size=batch_size, shuffle=False)

    print("\t* Loading test data...")
    with open(valid_mismatched_file, "rb") as pkl:
        valid_mismatched_data = pickle.load(pkl)
        valid_mismatched_dataloader = transform_batch_data(valid_mismatched_data, batch_size=batch_size, shuffle=False)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")

    model = ESIM(embedding_size,
                 hidden_size,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model,
                                             valid_matched_dataloader,
                                             criterion)
    print("\t* Matched Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))

    _, valid_loss, valid_accuracy = validate(model,
                                             valid_mismatched_dataloader,
                                             criterion)
    print("\t* Mismatched Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        train_dataloader = transform_batch_data(train_data, batch_size=batch_size, shuffle=True)

        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_dataloader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          valid_matched_dataloader,
                                                          criterion)

        valid_losses.append(epoch_loss)
        print("-> Matched Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          valid_mismatched_dataloader,
                                                          criterion)
        print("-> Mismatched Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        sys.stdout.flush() #刷新输出
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    fig = plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    fig.savefig('quora_loss.png')


if __name__ == "__main__":
    default_config = "../../config/training/mnli_training_bert.json"

    parser = argparse.ArgumentParser(
        description="Train the ESIM model on snli")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_dir = script_dir + '/scripts/training'

    parser.add_argument("--checkpoint",
                        default=None,#os.path.dirname(os.path.realpath(__file__)) + '/data/checkpoints/MNLI/bert/' +"esim_{}.pth.tar".format(1),
                        help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
         os.path.normpath(os.path.join(script_dir, config["valid_data_matched"])),
         os.path.normpath(os.path.join(script_dir, config["valid_data_mismatched"])),
         os.path.normpath(os.path.join(script_dir, config["target_dir"])),
         config["embedding_size"],
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         args.checkpoint)
