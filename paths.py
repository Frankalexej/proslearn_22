import os

root_ = "../"   # hyperpath of here
root_hd_ = '/mnt/storage/compling/proslearn/'    # hyperpath of harddisk
src_eng_ = root_hd_ + "/src/eng/"
src_man_ = root_hd_ + "/src/man_tone/"

debug_ = src_eng_ + "debug/"

model_save_ = root_ + "model_save/"



def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)