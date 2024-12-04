"""
B3: 
This time using recontruction as training goal (ConvAutoencoder), other things minimally changed compared to A3. 
This is Tone Learning. 
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

from H_1_models import ConvAutoencoder, LinearPredictor
from model_dataset import TokenMap
from model_dataset import ToneDatasetReconstruction as ThisDataset
from model_incremental import *
from model_trainer import ClusteringTrainer as ThisTrainer
# from model_filter import XpassFilter
from paths import *
from misc_recorder import *
# from H_2_drawer import draw_learning_curve_and_accuracy


def draw_learning_curve_and_accuracy(datas, data_names, epoch="", save=False, save_name=""): 
    plt.clf()
    assert len(datas) == len(data_names), "Data and Data Names should have the same length. "
    num_data = len(datas)
    fig, axs = plt.subplots(1, num_data, figsize=(6*num_data, 4))

    for i, data in enumerate(datas): 
        valid_data, full_valid_data = data
        ax = axs[i]

        ax.plot(valid_data, label='Valid')
        ax.plot(full_valid_data, label='Full Valid')
        ax.set_title("Learning Curve" + f"{data_names[i]} {epoch}")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)
    # plt.close()


def run_once_eval(hyper_dir_save, hyper_dir_read, model_type="large", pretype="f", posttype="f", sel="full", preepochs=20, postepochs=20, configs={}): 
    """
    Run once eval: 
    This is to run the training of linear classifier and evaluate the performance. 
    """

    model_save_dir = os.path.join(hyper_dir_save, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}")    # each folder represents one training epoch. 
    model_read_dir = os.path.join(hyper_dir_read, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}")    # for training savings, the epochs are just x.pt files. 
    mk(model_save_dir)
    mk(model_read_dir)
    guides_dir = os.path.join(hyper_dir_read, "guides") # only read from read dir

    # Loss Recording
    valid_vmeasures = ListRecorder(os.path.join(model_save_dir, "valid.vmeasure"))
    full_valid_vmeasures = ListRecorder(os.path.join(model_save_dir, "full_valid.vmeasure"))
    valid_aris = ListRecorder(os.path.join(model_save_dir, "valid.ari"))
    full_valid_aris = ListRecorder(os.path.join(model_save_dir, "full_valid.ari"))
    valid_nmis = ListRecorder(os.path.join(model_save_dir, "valid.nmi"))
    full_valid_nmis = ListRecorder(os.path.join(model_save_dir, "full_valid.nmi"))

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "convAE": 
        model_trained = ConvAutoencoder() # used to generate hidrep
        # model_eval = LinearPredictor() 
        # NOTE: this is just for evaluation, but it will have some training, which is not noted down, we only set a fix number of training epochs. 
    else:
        raise Exception("Model not defined! ")
    
    mylist = ["1", "2", "3", "4"]
    mymap = TokenMap(mylist)

    # Dataset Loaders
    # pool messanger
    pool_messanger = PoolMessanger(configs["num_dataset"], configs["data_type_mapper"][pretype], configs["data_type_mapper"][posttype], guides_dir)

    # NOTE: Subset Cache, this is to manage the reading of datasets. Should be transparent to user. 
    # NOTE: This is for training the classifier. 
    # train_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)
    valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)
    full_valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)

    # Learning Path Planner
    learning_plan_df = pd.read_csv(os.path.join(guides_dir, "learning_plan.csv"))
    learning_plan = learning_plan_df["dataset_id"].tolist()

    # Baseline & Train (I)
    base_epoch = 1
    for epoch in range(0, base_epoch + preepochs):  # NOTE: we have 0 epoch for baseline. 
        print(f"Epoch {epoch}")
        # Load model
        if the_epoch > 0: 
            model_name = "{}.pt".format(the_epoch)  # read from training savings the model to produce hidrep. 
            model_path = os.path.join(model_read_dir, model_name)   # NOTE: not the model_save_dir. 
            state = torch.load(model_path)
            model_trained.load_state_dict(state)
        # else we use the default initialization to mimic before-training baseline. 
        # Place model to device
        model_trained.to(device)
        # Trainer
        trainer = ThisTrainer(model_trained=model_trained, 
                            criterion=None, 
                            optimizer=None, 
                            model_save_dir=model_save_dir, 
                            n_clusters=4, # NOTE: this is the number if tones. 
                            device=device)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_vmeasure, valid_ari, valid_nmi = trainer.evaluate(valid_loader)
        valid_vmeasures.append(valid_vmeasure)
        valid_aris.append(valid_ari)
        valid_nmis.append(valid_nmi)

        # Full Validation Data
        dataset_id = learning_plan[epoch]
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="full_valid")
        full_valid_loader = full_valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        full_valid_vmeasure, full_valid_ari, full_valid_nmi = trainer.evaluate(full_valid_loader)
        full_valid_vmeasures.append(full_valid_vmeasure)
        full_valid_aris.append(full_valid_ari)
        full_valid_nmis.append(full_valid_nmi)


        if epoch % 10 == 0: 
            # do not save every epoch, but every 10 epochs. This saves time. 
            valid_vmeasures.save()
            valid_aris.save()
            valid_nmis.save()
            full_valid_vmeasures.save()
            full_valid_aris.save()
            full_valid_nmis.save()
            draw_learning_curve_and_accuracy(datas=[(valid_vmeasures.get(), full_valid_vmeasures.get()),
                                                    (valid_aris.get(), full_valid_aris.get()),
                                                    (valid_nmis.get(), full_valid_nmis.get())], 
                                            data_names=["VMeasure", "ARI", "NMI"], 
                                            epoch=str(epoch), 
                                            save=True, 
                                            save_name=f"{model_save_dir}/vis.png")

    # Train (II)
    base_epoch_II = base_epoch + preepochs
    pool_messanger.turn_on_full()   # IMPORTANT: turn on full data
    for epoch in range(base_epoch_II, base_epoch_II + postepochs):
        print(f"Epoch {epoch}")
        # Load model
        if the_epoch > 0: 
            model_name = "{}.pt".format(the_epoch)  # read from training savings the model to produce hidrep. 
            model_path = os.path.join(model_read_dir, model_name)   # NOTE: not the model_save_dir. 
            state = torch.load(model_path)
            model_trained.load_state_dict(state)
        # else we use the default initialization to mimic before-training baseline. 
        # Place model to device
        model_trained.to(device)
        # Trainer
        trainer = ThisTrainer(model_trained=model_trained, 
                            criterion=None, 
                            optimizer=None, 
                            model_save_dir=model_save_dir, 
                            n_clusters=4, # NOTE: this is the number if tones. 
                            device=device)

        # Validation Data
        dataset_id = learning_plan[epoch]   # this repetition serves for later multi-plan usage. 
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id,
                                                                            eval_type="valid")
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_vmeasure, valid_ari, valid_nmi = trainer.evaluate(valid_loader)
        valid_vmeasures.append(valid_vmeasure)
        valid_aris.append(valid_ari)
        valid_nmis.append(valid_nmi)

        # Full Validation Data
        full_valid_vmeasure = valid_vmeasure
        full_valid_ari = valid_ari
        full_valid_nmi = valid_nmi

        full_valid_vmeasures.append(full_valid_vmeasure)
        full_valid_aris.append(full_valid_ari)
        full_valid_nmis.append(full_valid_nmi)


        if epoch % 10 == 0: 
            # do not save every epoch, but every 10 epochs. This saves time. 
            valid_vmeasures.save()
            valid_aris.save()
            valid_nmis.save()
            full_valid_vmeasures.save()
            full_valid_aris.save()
            full_valid_nmis.save()
            draw_learning_curve_and_accuracy(datas=[(valid_vmeasures.get(), full_valid_vmeasures.get()),
                                                    (valid_aris.get(), full_valid_aris.get()),
                                                    (valid_nmis.get(), full_valid_nmis.get())], 
                                            data_names=["VMeasure", "ARI", "NMI"], 
                                            epoch=str(epoch), 
                                            save=True, 
                                            save_name=f"{model_save_dir}/vis.png")

    draw_learning_curve_and_accuracy(datas=[(valid_vmeasures.get(), full_valid_vmeasures.get()),
                                            (valid_aris.get(), full_valid_aris.get()),
                                            (valid_nmis.get(), full_valid_nmis.get())], 
                                    data_names=["VMeasure", "ARI", "NMI"], 
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
        "total_classifier_training_epochs": 20,
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
        train_name = "B3"
        model_save_dir = os.path.join(model_save_, f"{train_name}-cluster-{ts}")
        model_read_dir = os.path.join(model_save_, f"{train_name}-{ts}")
        print(f"{train_name}-{ts}")
        mk(model_save_dir)
        mk(model_read_dir)
        torch.cuda.set_device(args.gpu)
        run_once_eval(model_save_dir, model_read_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
                    preepochs=args.preepochs, postepochs=(configs["total_epochs"] - args.preepochs), configs=configs)
