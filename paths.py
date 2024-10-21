import os

root_ = "../"   # hyperpath of here
# using english as default, to facilitate code transferbnd_detect_tools
src_eng_ = root_ + "/src/eng/"
projects_ = "../../"
train_audio_ = projects_ + "wavln/src/eng/train-clean-100-audio/"
train_cut_syllable_ = src_eng_ + "train-clean-100-cs/"

debug_ = src_eng_ + "debug/"

model_save_ = root_ + "model_save/"



def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)