"""
A1: 
First run of prosody learning. This run is only on English, and the filter is just cut-off filter. 
A2: 
First run using pre-processed data. This should be also using gradual data feeding. 
This means that we need to write code for loading with multiple dataloaders (for simplicity, 
we are now just using dataloaders as a whole, each epoch should feed in one such dataloader)
and a function to plan learning process. 

Thanks to Ming, we can easily get the separated files from the full dataset. 
I guess for each run, we need to fix the data used, just like what we did before. 
Therefore, during data preparartion period, we will load the full data and get the 
corresponding tokens to be used and then we just save them separately for each dataset to access. 

A3: 
# This time using recontruction as training goal (ConvAutoencoder), other things minimally changed. 

--- Changed: A3 is now changed to run Mandarin tone prediction. 

A4: 
This is running simulated Mandarin tone dataset, which is generated from the original Mandarin dataset. 
"""

# All in Runner
## Importing the libraries
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torchinfo import summary
import torch.nn.functional as F
from torch.nn import init
import argparse

from H_1_models import SmallNetwork, MediumNetwork, LargeNetwork, ResLinearNetwork, LSTMNetwork, TwoConvNetwork
from model_dataset import TokenMap
from model_dataset import ToneDatasetNew as ThisDataset
from model_incremental import *
from model_trainer import ModelTrainer
# from model_filter import XpassFilter
from paths import *
from misc_recorder import *
from H_2_drawer import draw_learning_curve_and_accuracy


def draw_learning_curve_and_accuracy(losses, accs, epoch="", best_val=None, save=False, save_name=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, full_valid_losses = losses
    train_accs, valid_accs, full_valid_accs = accs

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.plot(full_valid_losses, label='Full Valid')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    ax2.plot(train_accs, label='Train')
    ax2.plot(valid_accs, label='Valid')
    ax2.plot(full_valid_accs, label='Full Valid')
    ax2.set_title('Learning Curve Accuracy' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)
    # plt.close()


def run_once_continue(hyper_dir, model_type="large", pretype="f", posttype="f", sel="full", preepochs=20, postepochs=20, configs={}): 
    """
    Frank's Note: 
    This is the function for running the training and simultaneous validation once. 
    Although it is to be called by a "main" function, this whole .py file only runs the training once. 
    Multiple runnings are implemented with multi-processing using bash scripts (.sh).

    Args: 
        hyper_dir (str): the uppermost saving directory under model_save_, named as runner_name-timestamp-run_number. 
        model_type (str): The type of model to be used. This includes "small", "medium", "large", "reslin", "lstm", etc. 
        pretype (str): The type of data to be used for the first phase of training. f=full, l=low, h=high. Default is "f".
        posttype (str): The type of data to be used for the second phase of training. Same as pretype. Default is "f".
        sel (str): Selected phoneme types, full, c=consonants, v=vowels. Default is "full". *Now only using full*
        preepochs (int): The number of epochs for the first phase of training. Default is 20.
        postepochs (int): The number of epochs for the second phase of training. Default is 20.
    """

    model_save_dir = os.path.join(hyper_dir, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}")
    guides_dir = os.path.join(hyper_dir, "guides")

    # Find the last epoch *saved* (N.B. we could only rely on saved *models*, we shall not rely on saved *losses* or *accuracies*)
    # Find from the model_save_dir the last model saved: the largest number ending with .pt
    dir_files = os.listdir(model_save_dir)
    model_files = [f for f in dir_files if f.endswith('.pt')]
    model_numbers = [int(f.split('.')[0]) for f in model_files if f.split('.')[0].isdigit()]
    ## Find the file with the largest number
    if model_numbers:
        last_epoch = max(model_numbers)
    else: 
        last_epoch = 0

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    full_valid_losses = ListRecorder(os.path.join(model_save_dir, "full_valid.loss"))
    train_accs = ListRecorder(os.path.join(model_save_dir, "train.acc"))
    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.acc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.acc"))
    ## Read the previous records
    train_losses.read_adjust_length(last_epoch)
    valid_losses.read_adjust_length(last_epoch)
    full_valid_losses.read_adjust_length(last_epoch)
    train_accs.read_adjust_length(last_epoch)
    valid_accs.read_adjust_length(last_epoch)
    full_valid_accs.read_adjust_length(last_epoch)

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    if model_type == "small":
        model = SmallNetwork()
    elif model_type == "medium":
        model = MediumNetwork()
    elif model_type == "large": 
        model = LargeNetwork()
    elif model_type == "reslin": 
        model = ResLinearNetwork()
    elif model_type == "lstm": 
        model = LSTMNetwork()
    elif model_type == "twoconvCNN": 
        model = TwoConvNetwork()        
    else:
        raise Exception("Model not defined! ")
    model.to(device)
    """
    Oops! It seems that to restore training without other turbulance, we need to restore the optimizer as well. 
    In this sense, we need to re-start everyting. 
    """
    optimizer = optim.Adam(model.parameters(), lr=configs["lr"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                                        mode='min', 
    #                                                        factor=0.1, 
    #                                                        patience=10, 
    #                                                        threshold=1e-4)
    # CosineAnnealingLR(optimizer, T_max=10)
    

    # Save Model Summary
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model, input_size=(128, 1, 128, 126))))
    
    mylist = ["0", "1", "2"]
    mymap = TokenMap(mylist)

    # Trainer
    trainer = ModelTrainer(model=model, 
                           criterion=criterion, 
                           optimizer=optimizer, 
                           model_save_dir=model_save_dir, device=device)
    # Dataset Loaders
    # pool messanger
    pool_messanger = PoolMessanger(configs["num_dataset"], configs["data_type_mapper"][pretype], configs["data_type_mapper"][posttype], guides_dir)

    # NOTE: Subset Cache, this is to manage the reading of datasets. Should be transparent to user. 
    train_cache = SubsetCache(max_cache_size=configs["max_cache_size_train"], dataset_class=ThisDataset)
    valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)
    full_valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)

    # Learning Path Planner
    planner = LearningPathPlanner(dataset_ids=pool_messanger.get_pool(), 
                                  total_epochs=configs["total_epochs"] + 1, # +1 because we have a pre-learning baseline. 
                                  p1=configs["lpp_configs"]["p1"], 
                                  decay_rate=configs["lpp_configs"]["decay_rate"])
    
    # generate the plan and save for reference
    learning_plan = planner.generate_learning_path()
    learning_plan_df = pd.DataFrame(learning_plan, columns=['dataset_id'])
    learning_plan_df.to_csv(os.path.join(model_save_dir, "learning_plan.csv"), index=False)
    """
    Currently we are using the same number of training and validation datasets. 
    TODO: In the future, if we want smaller number of validation datasets, we need to have two plans for training and validation independently. 
    """

    """Training"""
    """No Learning Baseline Get"""
    for epoch in range(0, 1):
        # Training Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id, 
                                                                            eval_type="train")
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.evaluate(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_loss, full_valid_acc = trainer.evaluate(full_valid_loader)
        full_valid_losses.append(full_valid_loss)
        full_valid_accs.append(full_valid_acc)
    train_losses.save()
    valid_losses.save()
    full_valid_losses.save()
    train_accs.save()
    valid_accs.save()
    full_valid_accs.save()

    # Train (I)
    base_epoch = 1

    for epoch in range(base_epoch, base_epoch + preepochs): 
        print(f"Epoch {epoch}")
        # Training Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id, 
                                                                            eval_type="train")
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.train(train_loader, epoch=epoch)  # this is the only difference. 
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_loss, full_valid_acc = trainer.evaluate(full_valid_loader)
        full_valid_losses.append(full_valid_loss)
        full_valid_accs.append(full_valid_acc)

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()

        if epoch % 10 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    # Train (II)
    base_epoch_II = base_epoch + preepochs
    pool_messanger.turn_on_full()   # turn on full data
    for epoch in range(base_epoch_II, base_epoch_II + postepochs):
        print(f"Epoch {epoch}")
        # Training Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id, 
                                                                            eval_type="train")
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.train(train_loader, epoch=epoch)  # this is the only difference. 
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_loss, full_valid_acc = trainer.evaluate(full_valid_loader)
        full_valid_losses.append(full_valid_loss)
        full_valid_accs.append(full_valid_acc)

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()

        if epoch % 10 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(base_epoch_II + postepochs), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")



def run_once(hyper_dir, model_type="large", pretype="f", posttype="f", sel="full", preepochs=20, postepochs=20, configs={}): 
    """
    Frank's Note: 
    This is the function for running the training and simultaneous validation once. 
    Although it is to be called by a "main" function, this whole .py file only runs the training once. 
    Multiple runnings are implemented with multi-processing using bash scripts (.sh).

    Args: 
        hyper_dir (str): the uppermost saving directory under model_save_, named as runner_name-timestamp-run_number. 
        model_type (str): The type of model to be used. This includes "small", "medium", "large", "reslin", "lstm", etc. 
        pretype (str): The type of data to be used for the first phase of training. f=full, l=low, h=high. Default is "f".
        posttype (str): The type of data to be used for the second phase of training. Same as pretype. Default is "f".
        sel (str): Selected phoneme types, full, c=consonants, v=vowels. Default is "full". *Now only using full*
        preepochs (int): The number of epochs for the first phase of training. Default is 20.
        postepochs (int): The number of epochs for the second phase of training. Default is 20.
    """

    model_save_dir = os.path.join(hyper_dir, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}")
    mk(model_save_dir)
    guides_dir = os.path.join(hyper_dir, "guides")

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    full_valid_losses = ListRecorder(os.path.join(model_save_dir, "full_valid.loss"))
    train_accs = ListRecorder(os.path.join(model_save_dir, "train.acc"))
    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.acc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.acc"))

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    if model_type == "small":
        model = SmallNetwork()
    elif model_type == "medium":
        model = MediumNetwork()
    elif model_type == "large": 
        model = LargeNetwork()
    elif model_type == "reslin": 
        model = ResLinearNetwork()
    elif model_type == "lstm": 
        model = LSTMNetwork()
    elif model_type == "twoconvCNN": 
        model = TwoConvNetwork(out_features=configs["output_dim"])
    else:
        raise Exception("Model not defined! ")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs["lr"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                                        mode='min', 
    #                                                        factor=0.1, 
    #                                                        patience=10, 
    #                                                        threshold=1e-4)
    # CosineAnnealingLR(optimizer, T_max=10)
    

    # Save Model Summary
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model, input_size=(128, 1, 128, 126))))
    
    mylist = ["1", "2", "3", "4"]
    mymap = TokenMap(mylist)

    # Trainer
    trainer = ModelTrainer(model=model, 
                           criterion=criterion, 
                           optimizer=optimizer, 
                           model_save_dir=model_save_dir, device=device)
    # Dataset Loaders
    # pool messanger
    pool_messanger = PoolMessanger(configs["num_dataset"], configs["data_type_mapper"][pretype], configs["data_type_mapper"][posttype], guides_dir)

    # NOTE: Subset Cache, this is to manage the reading of datasets. Should be transparent to user. 
    train_cache = SubsetCache(max_cache_size=configs["max_cache_size_train"], dataset_class=ThisDataset)
    valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)
    full_valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)

    # Learning Path Planner
    planner = LearningPathPlanner(dataset_ids=pool_messanger.get_pool(), 
                                  total_epochs=configs["total_epochs"] + 1, # +1 because we have a pre-learning baseline. 
                                  p1=configs["lpp_configs"]["p1"], 
                                  decay_rate=configs["lpp_configs"]["decay_rate"])
    
    # generate the plan and save for reference
    learning_plan = planner.generate_learning_path()
    learning_plan_df = pd.DataFrame(learning_plan, columns=['dataset_id'])
    learning_plan_df.to_csv(os.path.join(model_save_dir, "learning_plan.csv"), index=False)
    """
    Currently we are using the same number of training and validation datasets. 
    TODO: In the future, if we want smaller number of validation datasets, we need to have two plans for training and validation independently. 
    """

    """Training"""
    """No Learning Baseline Get"""
    for epoch in range(0, 1):
        # Training Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id, 
                                                                            eval_type="train")
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.evaluate(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_loss, full_valid_acc = trainer.evaluate(full_valid_loader)
        full_valid_losses.append(full_valid_loss)
        full_valid_accs.append(full_valid_acc)
    train_losses.save()
    valid_losses.save()
    full_valid_losses.save()
    train_accs.save()
    valid_accs.save()
    full_valid_accs.save()

    # Train (I)
    base_epoch = 1

    for epoch in range(base_epoch, base_epoch + preepochs): 
        print(f"Epoch {epoch}")
        # Training Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id, 
                                                                            eval_type="train")
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.train(train_loader, epoch=epoch)  # this is the only difference. 
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_loss, full_valid_acc = trainer.evaluate(full_valid_loader)
        full_valid_losses.append(full_valid_loss)
        full_valid_accs.append(full_valid_acc)

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()

        if epoch % 10 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    # Train (II)
    base_epoch_II = base_epoch + preepochs
    pool_messanger.turn_on_full()   # turn on full data
    for epoch in range(base_epoch_II, base_epoch_II + postepochs):
        print(f"Epoch {epoch}")
        # Training Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id, 
                                                                            eval_type="train")
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.train(train_loader, epoch=epoch)  # this is the only difference. 
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_loss, full_valid_acc = trainer.evaluate(full_valid_loader)
        full_valid_losses.append(full_valid_loss)
        full_valid_accs.append(full_valid_acc)

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()

        if epoch % 10 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(base_epoch_II + postepochs), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--dataprepare', '-dp', action="store_true")
    parser.add_argument('--traincontinue', '-ct', action="store_true")
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "large",help="Model type: small, medium, large, and others")
    parser.add_argument('--pretype','-p',type=str, default="f", help='Pretraining data type')
    parser.add_argument('--select','-s',type=str, default="full", help='Select full, consonants or vowels')
    parser.add_argument('--preepochs','-pree',type=int, default=20, help='Number of epochs in pre-training')
    parser.add_argument('--postepochs','-poste',type=int, default=20, help='Number of epochs in post-training')

    args = parser.parse_args()
    RUN_TIMES = 1

    configs = {
        "num_dataset": 50,
        "size_train": 1600, 
        "size_valid": 320,
        "data_type": "mel",
        "total_epochs": 300, # for lower pre-training conditions we lower the final epoch to 300, because we found no much learning afterwards. 
        "lr": 1e-4,
        "data_type_mapper": {
            "f": "full", 
            "l": "low",
            "h": "high"
        }, 
        "lpp_configs": {
            "p1": 0.5, 
            "decay_rate": 0.3
        }, 
        "max_cache_size_train": 7, 
        "max_cache_size_valid": 20, 
        "output_dim": 4, 
    }
    for run_time in range(RUN_TIMES):
        ## Hyper-preparations
        # ts = str(get_timestamp())
        ts = args.timestamp
        train_name = "A4"
        model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
        print(f"{train_name}-{ts}")
        mk(model_save_dir) 

        if args.dataprepare: 
            # Data Preparation
            guides_dir = os.path.join(model_save_dir, "guides") # for saving metadata and data
            mk(guides_dir)
            ### Get Data (Not Loading)
            mylist = ["1", "2", "3", "4"]   # the four tones in Mandarin
            mymap = TokenMap(mylist)

            dg_cons_train = ConstructDatasetGroup(
                src_path=src_man_, 
                all_meta_filename="meta_train.pkl", # NOTE: using csv this time, but csv and pkl are the same content. 
                                                    # NOTE: A4 using pkl, because provided. 
                target_dir=guides_dir, 
                target_name="train", 
                # pre_select={"stress_type": [1, 2, 3, 4]}
            )
            dg_cons_valid = ConstructDatasetGroup(
                src_path=src_man_, 
                all_meta_filename="meta_test.pkl",   # NOTE: we are actually using the test set as validation set, but it is fine, just switch. 
                target_dir=guides_dir,
                target_name="valid", 
                # pre_select={"stress_type": [1, 2, 3, 4]}
            )

            # # Construct and Save
            dg_cons_train.construct(
                num_dataset=configs["num_dataset"], 
                size_dataset=configs["size_train"], 
                absolute_size=True, 
                data_type=configs["data_type"], 
                select_column=['stress_type','index']
            )
            dg_cons_valid.construct(
                num_dataset=configs["num_dataset"],
                size_dataset=configs["size_valid"],
                absolute_size=True,
                data_type=configs["data_type"],
                select_column=['stress_type','index']
            )
            # Append Construct and Save
            # dg_cons_train.append_construct(
            #     num_dataset=configs["num_dataset"], 
            #     data_type=configs["data_type"], 
            #     append_pass="high"
            # )
            # dg_cons_valid.append_construct(
            #     num_dataset=configs["num_dataset"],
            #     data_type=configs["data_type"],
            #     append_pass="high"
            # )
        elif args.traincontinue: 
            raise Exception("Not implemented yet")
            # # we are restoring from interrupt
            # torch.cuda.set_device(args.gpu)

            # # for this mode, we shall detect whether go for continue or go for new. 
            # model_save_dir = os.path.join(model_save_dir, f"{args.model}-{args.preepochs}-{(configs["total_epochs"] - args.preepochs)}", args.select, f"{args.pretype}f")
            # if os.path.exists(model_save_dir): 
            #     run_once_continue(model_save_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
            #                     preepochs=args.preepochs, postepochs=(configs["total_epochs"] - args.preepochs), configs=configs)
            # else: 
            #     run_once(model_save_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
            #                 preepochs=args.preepochs, postepochs=(configs["total_epochs"] - args.preepochs), configs=configs)
        else: 
            torch.cuda.set_device(args.gpu)
            run_once(model_save_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
                        preepochs=args.preepochs, postepochs=(configs["total_epochs"] - args.preepochs), configs=configs)
