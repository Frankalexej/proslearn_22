"""
H20: 
This runner will try to run this on multiple models: CNN(current), RNN and Linear. 
Since the running logic is all the same, and the only difference lies in the model structure, 
we mainly change the model, while keeping the in and outs all the same. 
H21: 
This is staged running. We want to test the effect of number of "pretraining" epochs. 

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

from H_1_models import SmallNetwork, MediumNetwork, LargeNetwork, ResLinearNetwork, LSTMNetwork
from model_configs import ModelDimConfigs, TrainingConfigs
from misc_tools import get_timestamp, ARPABET
from model_dataset import DS_Tools, Padder, TokenMap, NormalizerKeepShape
from model_dataset import SyllableDatasetNew as ThisDataset
from model_dataset import ConstructDatasetGroup
from model_filter import XpassFilter
from paths import *
from misc_progress_bar import draw_progress_bar
from misc_recorder import *
from H_2_drawer import draw_learning_curve_and_accuracy



# Data Loader
def load_data(type="f", sel="full", load="train"):
    if type == "l":
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            XpassFilter(cut_off_upper=500),
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
        )
    elif type == "h": 
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            XpassFilter(cut_off_upper=10000, cut_off_lower=4000),
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
        )
    else: 
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
        )
    # with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
    #     # Load the object from the file
    #     mylist = pickle.load(file)
    #     mylist.remove('AH') # we don't include this, it is too mixed. 

    select = ["0", "1", "2"]
    mymap = TokenMap(select)
    # if sel == "c": 
    #     select = ARPABET.intersect_lists(mylist, ARPABET.list_consonants())
    # elif sel == "v":
    #     select = ARPABET.intersect_lists(mylist, ARPABET.list_vowels())
    # else:
    #     select = mylist
    # Now you can use the loaded object
    # mymap = TokenMap(mylist)
    if load == "train": 
        train_ds = ThisDataset(train_cut_syllable_, 
                            os.path.join(src_eng_, "guide_train_mod.csv"), 
                            select=select, 
                            mapper=mymap, 
                            transform=mytrans)
        
        train_ds_indices = DS_Tools.read_indices(os.path.join(model_save_dir, f"train_{sel}.use"))
        use_train_ds = torch.utils.data.Subset(train_ds, train_ds_indices)
        train_loader = DataLoader(use_train_ds, batch_size=TrainingConfigs.BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=TrainingConfigs.LOADER_WORKER)
        
        return train_loader
    elif load == "valid":
        valid_ds = ThisDataset(train_cut_syllable_, 
                            os.path.join(src_eng_, "guide_validation_mod.csv"), 
                            select=select, 
                            mapper=mymap,
                            transform=mytrans)
        valid_ds_indices = DS_Tools.read_indices(os.path.join(model_save_dir, f"valid_{sel}.use"))
        use_valid_ds = torch.utils.data.Subset(valid_ds, valid_ds_indices)
        valid_loader = DataLoader(use_valid_ds, batch_size=TrainingConfigs.BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=TrainingConfigs.LOADER_WORKER)
        return valid_loader

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

def run_once(hyper_dir, model_type="large", pretype="f", posttype="f", sel="full", preepochs=20, postepochs=20): 
    """
    Frank's Note: 
    This is the function for running the training and simultaneous validation once. 
    Although it is to be called by a "main" function, this whole .py file only runs the training once. 
    Multiple runnings are implemented with multi-processing using bash scripts (.sh).

    Parameters: 
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

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    full_valid_losses = ListRecorder(os.path.join(model_save_dir, "full_valid.loss"))
    trainlikevalid_losses = ListRecorder(os.path.join(model_save_dir, "trainlikevalid.loss"))

    train_accs = ListRecorder(os.path.join(model_save_dir, "train.acc"))
    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.acc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.acc"))
    trainlikevalid_accs = ListRecorder(os.path.join(model_save_dir, "trainlikevalid.acc"))

    special_recs = DictRecorder(os.path.join(model_save_dir, "special.hst"))

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
    else:
        raise Exception("Model not defined! ")
    # model= nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model, input_size=(128, 1, 64, 21))))

    # Load Data (I&II)
    train_loader_1 = load_data(type=pretype, sel="full", load="train")
    valid_loader_1 = load_data(type=pretype, sel=sel, load="valid") # target 
    train_loader_2 = load_data(type=posttype, sel="full", load="train")
    valid_loader_2 = load_data(type=posttype, sel=sel, load="valid")    # full = trainlike (because this time we don't separate c/v)
    # trainlikevalid_loader_1 = load_data(type=pretype, sel="full", load="valid")
    # trainlikevalid_loader_2 = load_data(type=posttype, sel="full", load="valid")
    # In this way, we get training data will both consonants and vowels, but validation data with only either consonants or vowels. 
    # But the sound range always follows the pretype and posttype settings. 

    # this is mainly to get the "improvement" for 
    # only-full training models, because they naturally
    # don't have a "transition" from nothing to 
    # "having been trained on full"
    """No Learning Baseline Get"""
    # Target Eval
    model.eval()
    valid_loss = 0.
    valid_num = len(valid_loader_1)
    valid_correct = 0
    valid_total = 0
    for idx, (x, y) in enumerate(valid_loader_1):
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)
        valid_loss += loss.item()

        pred = model.predict_on_output(y_hat)

        valid_total += y_hat.size(0)
        valid_correct += (pred == y).sum().item()

    special_recs.append(("notrain-target-loss", valid_loss / valid_num))
    special_recs.append(("notrain-target-acc", valid_correct / valid_total))
    special_recs.save()

    # Full Eval
    model.eval()
    full_valid_loss = 0.
    full_valid_num = len(valid_loader_2)
    full_valid_correct = 0
    full_valid_total = 0
    for idx, (x, y) in enumerate(valid_loader_2):
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)
        full_valid_loss += loss.item()

        pred = model.predict_on_output(y_hat)

        full_valid_total += y_hat.size(0)
        full_valid_correct += (pred == y).sum().item()

    special_recs.append(("notrain-full-loss", full_valid_loss / full_valid_num))
    special_recs.append(("notrain-full-acc", full_valid_correct / full_valid_total))
    special_recs.save()

    # Train (I)
    best_valid_loss = 1e9
    best_valid_loss_epoch = 0
    BASE = 0

    for epoch in range(BASE, BASE + preepochs):
        model.train()
        train_loss = 0.
        train_num = len(train_loader_1)    # train_loader
        train_correct = 0
        train_total = 0
        for idx, (x, y) in enumerate(train_loader_1):
            optimizer.zero_grad()
            x = x.to(device)
            # y = torch.tensor(y, device=device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            pred = model.predict_on_output(y_hat)
            train_total += y_hat.size(0)
            train_correct += (pred == y).sum().item()
            # draw_progress_bar(idx, train_num, title="Train")

        train_losses.append(train_loss / train_num)
        train_accs.append(train_correct / train_total)
        last_model_name = f"{epoch}.pt"
        torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        # Target Eval
        model.eval()
        valid_loss = 0.
        valid_num = len(valid_loader_1)
        valid_correct = 0
        valid_total = 0
        for idx, (x, y) in enumerate(valid_loader_1):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            valid_loss += loss.item()

            pred = model.predict_on_output(y_hat)

            valid_total += y_hat.size(0)
            valid_correct += (pred == y).sum().item()

        avg_valid_loss = valid_loss / valid_num
        valid_losses.append(avg_valid_loss)
        valid_accs.append(valid_correct / valid_total)
        if avg_valid_loss < best_valid_loss: 
            best_valid_loss = avg_valid_loss
            best_valid_loss_epoch = epoch

        # Full Eval
        model.eval()
        full_valid_loss = 0.
        full_valid_num = len(valid_loader_2)
        full_valid_correct = 0
        full_valid_total = 0
        for idx, (x, y) in enumerate(valid_loader_2):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            full_valid_loss += loss.item()

            pred = model.predict_on_output(y_hat)

            full_valid_total += y_hat.size(0)
            full_valid_correct += (pred == y).sum().item()

        full_valid_losses.append(full_valid_loss / full_valid_num)
        full_valid_accs.append(full_valid_correct / full_valid_total)

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

    # draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
    #                                 accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
    #                                 epoch=str(BASE + preepochs - 1), 
    #                                 save=True, 
    #                                 save_name=f"{model_save_dir}/vis.png")
    
    # Pre Model Best
    special_recs.append(("preval_epoch", best_valid_loss_epoch))
    special_recs.save()

    # Train (II)
    BASE = BASE + preepochs
    for epoch in range(BASE, BASE + postepochs):
        model.train()
        train_loss = 0.
        train_num = len(train_loader_2)    # train_loader
        train_correct = 0
        train_total = 0
        for idx, (x, y) in enumerate(train_loader_2):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            pred = model.predict_on_output(y_hat)
            train_total += y_hat.size(0)
            train_correct += (pred == y).sum().item()
            # draw_progress_bar(idx, train_num, title="Train")

        train_losses.append(train_loss / train_num)
        train_accs.append(train_correct / train_total)
        last_model_name = f"{epoch}.pt"
        torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        # Target Eval
        model.eval()
        valid_loss = 0.
        valid_num = len(valid_loader_2)
        valid_correct = 0
        valid_total = 0
        for idx, (x, y) in enumerate(valid_loader_2):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            valid_loss += loss.item()

            pred = model.predict_on_output(y_hat)

            valid_total += y_hat.size(0)
            valid_correct += (pred == y).sum().item()


        avg_valid_loss = valid_loss / valid_num
        valid_losses.append(avg_valid_loss)
        full_valid_losses.append(avg_valid_loss)
        valid_accs.append(valid_correct / valid_total)
        full_valid_accs.append(valid_correct / valid_total)
        if avg_valid_loss < best_valid_loss: 
            best_valid_loss = avg_valid_loss
            best_valid_loss_epoch = epoch

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
                                    epoch=str(BASE + postepochs - 1), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")
    
    # Post Model Best
    special_recs.append(("postval_epoch", best_valid_loss_epoch))
    special_recs.save()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--dataprepare', '-dp', action="store_true")
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "large",help="Model type: small, medium, large, and others")
    parser.add_argument('--pretype','-p',type=str, default="f", help='Pretraining data type')
    parser.add_argument('--select','-s',type=str, default="full", help='Select full, consonants or vowels')
    parser.add_argument('--preepochs','-pree',type=int, default=20, help='Number of epochs in pre-training')
    parser.add_argument('--postepochs','-poste',type=int, default=20, help='Number of epochs in post-training')

    args = parser.parse_args()
    RUN_TIMES = 1
    for run_time in range(RUN_TIMES):
        ## Hyper-preparations
        # ts = str(get_timestamp())
        ts = args.timestamp
        train_name = "A2"
        model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
        print(f"{train_name}-{ts}")
        mk(model_save_dir) 

        if args.dataprepare: 
            # Data Preparation
            guides_dir = os.path.join(model_save_dir, "guides") # for saving metadata and data
            mk(guides_dir)
            ### Get Data (Not Loading)
            mylist = ["0", "1", "2"]
            mymap = TokenMap(mylist)

            dg_cons_train = ConstructDatasetGroup(
                all_meta_path=os.path.join(src_eng_, "guide_train_syllableInfor.pkl"),
                target_dir=guides_dir, 
                target_name="train"
            )
            dg_cons_valid = ConstructDatasetGroup(
                all_meta_path=os.path.join(src_eng_, "guide_test_syllableInfor.pkl"),   # NOTE: we are actually using the test set as validation set, but it is fine, just switch. 
                target_dir=guides_dir,
                target_name="valid"
            )

            # Construct and Save
            dg_cons_train.construct(
                num_dataset=50, 
                num_per_dataset=1600, 
                absolute_size=True, 
                data_type="mel", 
                select_column=['stress_type','index']
            )
            dg_cons_valid.construct(
                num_dataset=50,
                num_per_dataset=320,
                absolute_size=True,
                data_type="mel",
                select_column=['stress_type','index']
            )
        else: 
            # TODO: not finished changing. 
            torch.cuda.set_device(args.gpu)
            run_once(model_save_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
                        preepochs=args.preepochs, postepochs=(40 - args.preepochs))
