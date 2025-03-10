import os
import torchaudio
import pandas as pd
from paths import *
from misc_tools import PathUtils as PU, AudioCut
from misc_multiprocessing import *
import argparse

# xxx_ means xxx_path (self-defined abbr)
def segment_sentence(au_, ca_, name, starttimes, endtimes, suids): 
    # in current setting save_small is always true
    audio_file = os.path.join(au_, name + ".flac")

    speaker, rec, sentence = AudioCut.solve_name(name)

    ca_name_ = os.path.join(ca_, sentence)
    PU.mk(ca_name_)

    for idx, (starttime, endtime, suid) in enumerate(zip(starttimes, endtimes, suids)): 
        try: 
            # Extract and save audio segment
            waveform, sample_rate = torchaudio.load(audio_file)
            start_sample = AudioCut.time2frame(starttime, sample_rate)
            end_sample = AudioCut.time2frame(endtime, sample_rate)
            cut_audio = waveform[:, start_sample:end_sample]

            ca_file = os.path.join(ca_name_, suid + ".flac")
            torchaudio.save(ca_file, cut_audio, sample_rate)
        except Exception: 
            pass
    return None

def segment(work_list, df, dir_au, dir_ca): 
    """
    This function reads textgrid files, and according to the 
    interval boundaries cut the corresponding recordings into small audios. 
    In the mean time note down the metadata needed for training. 
    """
    assert (PU.path_exist(dir_au) \
                and PU.path_exist(dir_ca))  # check dir existence
    
    for speaker in work_list: 
        print(speaker)
        file_names = df[df["speaker"] == speaker]["file"].unique().tolist()
    
        for file_name in file_names: 
            speaker, rec, sentence = AudioCut.solve_name(file_name)
            audio_speaker_rec_ = os.path.join(dir_au, speaker, rec)
            ca_speaker_rec_ = os.path.join(dir_ca, speaker, rec)

            PU.mk(ca_speaker_rec_)

            file_name_filtered_df = df[df["file"] == file_name]
            starttimes = file_name_filtered_df["syllable_startTime"].tolist()
            endtimes = file_name_filtered_df["syllable_endTime"].tolist()
            suids = file_name_filtered_df["suid"].tolist()

            segment_sentence(
                                au_=audio_speaker_rec_, 
                                ca_=ca_speaker_rec_, 
                                name=file_name, 
                                starttimes=starttimes, 
                                endtimes=endtimes, 
                                suids=suids
                            )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    # parser.add_argument('--level', '-l', type=str, default="words", help="Cut into words or phones")
    parser.add_argument('--set', '-s', type=str, default="train", help="Cut which set, train, validation or test. ")
    parser.add_argument('--num_processes', '-np', type=int, default=64, help="Number of processes")
    args = parser.parse_args()
    guide_df = pd.read_csv(os.path.join(src_eng_, f"guide_{args.set}.csv"))
    run_mp(segment, 
           guide_df["speaker"].unique().tolist(), args.num_processes, *(guide_df, train_audio_, train_cut_syllable_))

    # segment([1069], guide_df, train_audio_, train_cut_syllable_)