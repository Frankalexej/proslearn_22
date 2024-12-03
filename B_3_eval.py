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
from model_trainer import LinearClassifierPredictionTrainer as ThisTrainer
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


def run_once_eval(hyper_dir_save, hyper_dir_read, model_type="large", pretype="f", posttype="f", sel="full", preepochs=20, postepochs=20, the_epoch=0, configs={}): 
    """
    Run once eval: 
    This is to run the training of linear classifier and evaluate the performance. 
    """

    model_save_dir = os.path.join(hyper_dir_save, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}", f"{the_epoch}")    # each folder represents one training epoch. 
    model_read_dir = os.path.join(hyper_dir_read, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}")    # for training savings, the epochs are just x.pt files. 
    mk(model_save_dir)
    mk(model_read_dir)
    guides_dir = os.path.join(hyper_dir_read, "guides") # only read from read dir

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    full_valid_losses = ListRecorder(os.path.join(model_save_dir, "full_valid.loss"))
    train_accs = ListRecorder(os.path.join(model_save_dir, "train.acc"))
    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.acc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.acc"))

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()    # NOTE: CrossEntropyLoss is used for prediction. 
    if model_type == "convAE": 
        model_trained = ConvAutoencoder() # used to generate hidrep
        model_eval = LinearPredictor() 
        # NOTE: this is just for evaluation, but it will have some training, which is not noted down, we only set a fix number of training epochs. 
    else:
        raise Exception("Model not defined! ")
    
    # Load model
    if the_epoch > 0: 
        model_name = "{}.pt".format(the_epoch)  # read from training savings the model to produce hidrep. 
        model_path = os.path.join(model_read_dir, model_name)   # NOTE: not the model_save_dir. 
        state = torch.load(model_path)
        model_trained.load_state_dict(state)
    # else we use the default initialization to mimic before-training baseline. 

    # Place model to device
    model_trained.to(device)
    model_eval.to(device)
    optimizer = optim.Adam(model_eval.parameters(), lr=configs["lr"])

    # Save Model Summary
    model_str = str(model_eval)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model_eval, input_size=(128, 32, 32, 31))))
    
    mylist = ["1", "2", "3", "4"]
    mymap = TokenMap(mylist)

    # Trainer
    trainer = ThisTrainer(model_trained=model_trained, model_eval=model_eval,
                           criterion=criterion, 
                           optimizer=optimizer, 
                           model_save_dir=model_save_dir, device=device)
    # Dataset Loaders
    # pool messanger
    pool_messanger = PoolMessanger(configs["num_dataset"], configs["data_type_mapper"][pretype], configs["data_type_mapper"][posttype], guides_dir)

    # NOTE: Subset Cache, this is to manage the reading of datasets. Should be transparent to user. 
    # NOTE: This is for training the classifier. 
    train_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)
    valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)
    full_valid_cache = SubsetCache(max_cache_size=configs["max_cache_size_valid"], dataset_class=ThisDataset)

    # Learning Path Planner
    # planner = LearningPathPlanner(dataset_ids=pool_messanger.get_pool(), 
    #                               total_epochs=configs["total_epochs"] + 1, # +1 because we have a pre-learning baseline. 
    #                               p1=configs["lpp_configs"]["p1"], 
    #                               decay_rate=configs["lpp_configs"]["decay_rate"])
    
    # generate the plan and save for reference
    # learning_plan = planner.generate_learning_path()
    # learning_plan_df = pd.DataFrame(learning_plan, columns=['dataset_id'])
    learning_plan_df = pd.read_csv(os.path.join(guides_dir, "learning_plan.csv"))
    learning_plan = learning_plan_df["dataset_id"].tolist()
    learning_plan_pickuper = LearningPathPickup(learning_path=learning_plan, 
                                                dataset_pool=pool_messanger.get_pool(), 
                                                on_same="random", on_end="random")

    """Training and Evaluating Classifier"""
    for epoch in range(0, configs["total_classifier_training_epochs"]):  
        print(f"Epoch {epoch} of {the_epoch}")
        # Get dataset id
        dataset_id_train, dataset_id_eval = learning_plan_pickuper.get_this_and_next(epoch)

        # Training Data
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id_train, 
                                                                            eval_type="valid")  # we use valid as training data.
        train_loader = train_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        train_loss, train_acc = trainer.train(train_loader, epoch=epoch)  # this is the only difference. 
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Data
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id_eval,
                                                                            eval_type="valid")  # we use THE NEXT valid as validation data.
        valid_loader = valid_cache.get_subset(dataset_id, meta_path, data_path, mymap)
        valid_loss, valid_acc = trainer.evaluate(valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Full Validation Data
        dataset_id, meta_path, data_path = pool_messanger.get_loading_params(dataset_id_eval,
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

    draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(configs["total_classifier_training_epochs"]), 
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
        model_save_dir = os.path.join(model_save_, f"{train_name}-eval-{ts}")
        model_read_dir = os.path.join(model_save_, f"{train_name}-{ts}")
        print(f"{train_name}-{ts}")
        mk(model_save_dir)
        mk(model_read_dir)
        torch.cuda.set_device(args.gpu)
        for the_epoch in range(0, configs["total_epochs"]): 
            # we loop over this
            run_once_eval(model_save_dir, model_read_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
                        preepochs=args.preepochs, postepochs=(configs["total_epochs"] - args.preepochs), the_epoch=the_epoch, configs=configs)
