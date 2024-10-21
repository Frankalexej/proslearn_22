# 1/16/2024
# dataPreprocess.py-test.py
# compling
# description: This script is to manipulate array or list like datasets
# input:
# output:

import csv
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import concurrent.futures

def save_multiple_arrays(files:list,paths:list):
    '''
    save a list of pkl file based on a list of paths
    :param files: np.array
    :param paths: file path
    :return: None
    '''
    for file,filepath in zip(files,paths):
        np.save(filepath,file)

def process_in_chunks(func, data:list, chunk_size:int) -> list:
    '''
    do process in chuncks
    :param func: function
    :param data: a list of data
    :param chunk_size: how many data to process each time
    :return: processed data
    '''
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Process data in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            results.extend(executor.map(func, chunk))
    return results

def load_dataset(filename:str):
    '''
    This will load data from a pickle file
    :param filename: pickle file name
    :return: the data stored in pickle file
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_dataset(data,filename:os.PathLike):
    '''
    This will save data to a pickle file
    :param data: data to store in the pickle file
    :param filename: result path
    :return: None
    '''
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def save_dict_to_csv(dictionary: dict, file_path: os.PathLike):
    """
    Save a dict to csv file
    :param dictionary: record
    :param file_path: output file path
    :return: None
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(dictionary.keys())
        writer.writerow(dictionary.values())

def save_dicts_to_csv(list_of_dicts: list, file_path: os.PathLike):
    '''
    This will save a list of dicts with same structure to a csv file
    :param list_of_dicts: dictionaries with the same keys
    :param file_path: output file path
    :return: None
    '''
    fieldnames = list(list_of_dicts[0].keys())

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(list_of_dicts)

def checkDistribution(data:list, save ='') -> None:
    '''
    This will show the distribution of values in a list and draw pie plot for it
    :param data: the data
    :return: None
    '''

    # Count the occurrences of each class
    class_counts = Counter(data)
    # Print the class distribution
    # print(f"Num of classes: {len(class_counts.items())}")
    # for class_name, count in class_counts.items():
    #     print(f"Class '{class_name}': {count} occurrences")
    # Get the class names and counts
    class_names = list(class_counts.keys())
    class_values = list(class_counts.values())

    # Create a pie plot
    plt.pie(class_values, labels=class_names, autopct='%1.1f%%')
    # Set aspect ratio as equal to ensure a circular pie
    plt.axis('equal')
    # Set a title for the plot
    plt.title('Class Distribution')
    # Display the plot
    if save != '':
        with open(f'losses/cluster/distribution/{os.path.basename(save)}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(class_names,class_values))
        plt.savefig(save)
    plt.show()



def dict2list(original_dict: dict) -> list:
    '''
    Convert dict of lists to list of dict
    :param original_dict: each key value should be a list with the same length
    :return: a list of dicts
    '''
    # Get the keys of the original dictionary
    keys = list(original_dict.keys())

    # Transform the dictionary into a list of dictionaries
    transformed_list = []
    num_samples = len(original_dict[keys[0]])  # Assuming all values have the same length

    for i in range(num_samples):
        new_dict = {key: original_dict[key][i] for key in keys}
        transformed_list.append(new_dict)
    return transformed_list


def list2dict(original_list: list) -> dict:
    '''
    Convert a list of dicts to dict of lists
    :param original_list: a list of dicts
    :return: a dict of lists with the same format as orginal dicts
    '''
    # Get the keys of the original dictionary
    keys = list(original_list[0].keys())

    # Transform the list into a dictionary of lists
    transformed_dict = []
    for key in keys:
        transformed_dict[key] = [data[key] for data in original_list]
    return transformed_dict

def divideDataset(dataset:list, val_size:float,test_size:float,sortkey:str): # split dataset while keep the distribution of phones
    '''
    This function can split the full dataset into training,validation and testing set while keep the distribution of original dataset
    :param dataset: this is the full dataset
    :param val_size: specify the portions of training-validation
    :param test_size: specify the portions of (training&validation)-testing
    :param sortkey: the key on which you want to keep the distribution
    :return: training_set,validation_set,testing_set
    '''
    # Extract the key labels
    labels = [data[sortkey] for data in dataset]
    # Split the dataset into training and temporary sets
    train_temp, test = train_test_split(dataset, test_size=test_size, stratify=labels)
    # Split the temporary set into training and validation sets
    train, val = train_test_split(train_temp, test_size=val_size, stratify=[data[sortkey] for data in train_temp])

    return train,val,test


def pad_data_random_paired(audio_data: list, video_data: list, audio_target_shape: tuple, video_target_shape: tuple):
    """
        Pad the data to a specified target shape.
        If the input shape is larger than target, the input will be cropped.
        Args:
            data (numpy.ndarray): Input data of shape (n_samples, ...)
            target_shape (tuple): Target shape (target_length, target_width) for padding the data.

        Returns:
            numpy.ndarray: Padded data with shape (n_samples, target_length, target_width, ...)
        """
    n_tokens = len(audio_data)
    padded_audios = []
    padded_videos = []
    for n_sample in range(0, n_tokens):
        audio = audio_data[n_sample]
        video = video_data[n_sample]

        audio_pad_length = max(0, audio_target_shape[1] - audio.shape[1])
        video_pad_length = max(0, video_target_shape[1] - video.shape[1])
        pad_point = np.random.rand()
        if audio_pad_length > 0:
            audio_pad_point = max(0, int(audio_pad_length * pad_point) - 1)
            padded_audio = np.zeros((audio_target_shape[0], audio_target_shape[1]))
            padded_audio[:audio.shape[0], audio_pad_point:audio_pad_point + audio.shape[1]] = audio
        else:
            padded_audio = audio[:, :audio_target_shape[1]]
        if video_pad_length > 0:
            video_pad_point = max(0, int(video_pad_length * pad_point) - 1)
            padded_video = np.zeros((video_target_shape[0], video_target_shape[1]), dtype='uint8')
            padded_video[:video.shape[0], video_pad_point:video_pad_point + video.shape[1]] = video
        else:
            padded_video = video[:, :video_target_shape[1]]

        padded_audios.append(padded_audio)
        padded_videos.append(padded_video)

    return np.array(padded_audios), np.array(padded_videos)

def pad_data(data:list, target_shape:tuple):
    """
    Pad the data to a specified target shape.
    If the input shape is larger than target, the input will be cropped.
    Args:
        data (numpy.ndarray): Input data of shape (n_samples, ...)
        target_shape (tuple): Target shape (target_length, target_width) for padding the data.

    Returns:
        numpy.ndarray: Padded data with shape (n_samples, target_length, target_width, ...)
    """
    padded_data = []
    for sample in data:
        if sample.ndim == 1:  # Audio data
            pad_length = max(0, target_shape[0] - len(sample))
            if pad_length > 0:
                padded_sample = np.pad(sample, (0, pad_length), mode='constant')
            else:
                padded_sample = sample[:target_shape[0]]
        elif sample.ndim == 2:  # Image data
            pad_height = max(0, target_shape[0] - sample.shape[0])
            pad_width = max(0, target_shape[1] - sample.shape[1])
            if pad_width > 0:
                pad_values = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (sample.ndim - 2)
                padded_sample = np.pad(sample, pad_values, mode='constant')
            else:
                padded_sample = sample[:,:target_shape[1]]
        else:
            raise ValueError("Unsupported data dimension. Only 1D (audio) and 2D (image) data are supported.")

        padded_data.append(padded_sample)

    return np.array(padded_data)


def min_max_normalization(data: list):

    """
    Perform min-max normalization on the given dataset using scikit-learn's MinMaxScaler.

    Args:
        data (list): Input data of shape (n_samples, n_features).

    Returns:
        list: Normalized data with values scaled between 0 and 1, of the same shape as the input data.
    """

    normalized_data = []
    scaler = MinMaxScaler()

    for sample in data:
        if sample.ndim == 1:
            normalized_sample = scaler.fit_transform(sample.reshape(-1, 1))
            normalized_data.append(normalized_sample.flatten())
        else:
            # Convert the image to float type to handle decimal values
            sample = sample.astype(np.float32)
            # Scale the pixel values to the range [0, 1]
            normalized_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
            normalized_data.append(normalized_sample)
    return normalized_data

def concat_audio_video(audio_data: np.ndarray, video_data:np.ndarray):
    '''
    This is to concate audiodata and videodata, they should have the same size
    :param audio_data: (n_samples,spectrogram_heigth,spectrogram_width,channel)
    :param video_data: (n_samples,video_heigth,video_width,channel)
    :return: ((n_samples,features,channel)
    '''
    # Check if the dimensions of audio and video data match
    if audio_data.shape[0] != video_data.shape[0]:
        raise ValueError("Mismatch in the number of samples between audio and video data.")


    # Reshape audio and video data
    audio_data_reshaped = np.reshape(audio_data, (audio_data.shape[0],-1, audio_data.shape[-1]))
    video_data_reshaped = np.reshape(video_data, (video_data.shape[0], -1, audio_data.shape[-1]))

    # Concatenate audio and video data along the feature axis
    dataset = np.concatenate((audio_data_reshaped, video_data_reshaped), axis=2)


    return dataset
def data_generator_single(dataset, total_samples,datakey:str,labelkey:str,batch_size=32):
    '''
    data constructor for single modality
    :param dataset:
    :param total_samples:
    :param datakey: the input data key
    :param labelkey: the output data key
    :param batch_size:
    :return: a constructor yields batch_samples, batch_labels
    '''
    while True:
        try:
            # Generate batches of data
            for i in range(0, total_samples, batch_size):
                batch_samples = dataset[datakey][i:i + batch_size]
                batch_labels = dataset[labelkey][i:i + batch_size]
                yield batch_samples, batch_labels
        except EOFError:
                break

def data_generator_multi(total_samples, dataset,inputkey1, inputkey2, labelkey1,
                       labelkey2, batch_size = 32, placehold = False,hidden_dimension=256):
    '''
    data generator for multi data modalities
    :param batch_size:
    :param total_samples:
    :param dataset:  e.g. traningdata
    :param inputkey1: e.g. "audios"
    :param inputkey2:  e.g. "videos"
    :param labelkey1:  e.g. "labels"
    :param labelkey2:
    :param hidden_dimension: for placeholder
    :param placehold: if True, yield placeholder in labels
    :return: yield [batch_input1_samples, batch_input2_samples], [batch_labels2, batch_labels2,
                                                               (placeholder_cor)]
    '''
    while True:
        # Generate batches of data
        for i in range(0, total_samples, batch_size):
            batch_input1_samples = dataset[inputkey1][i:i + batch_size]
            batch_labels1 = dataset[labelkey1][i:i + batch_size]
            batch_input2_samples = dataset[inputkey2][i:i + batch_size]
            batch_labels2 = dataset[labelkey2][i:i + batch_size]
            if placehold is True:
                placeholder_cor = np.random.rand(*(batch_size, hidden_dimension * 2))
                yield [batch_input1_samples, batch_input2_samples], [batch_labels1, batch_labels2,
                                                               placeholder_cor]
            else:
                yield [batch_input1_samples, batch_input2_samples], [batch_labels1, batch_labels2]

def data_generator_multi_class(total_samples, dataset,inputkey1, inputkey2, labelkey, batch_size = 32):
    '''
    data generator for multi data modalities with labels as output
    :param batch_size:
    :param total_samples:
    :param dataset:  e.g. traningdata
    :param inputkey1: e.g. "audios"
    :param inputkey2:  e.g. "videos"
    :param labelkey:  e.g. "labels"
    :return: yield [batch_input1_samples, batch_input2_samples], batch_labels
    '''
    while True:
        # Generate batches of data
        for i in range(0, total_samples, batch_size):
            batch_input1_samples = dataset[inputkey1][i:i + batch_size]
            batch_labels = dataset[labelkey][i:i + batch_size]
            batch_input2_samples = dataset[inputkey2][i:i + batch_size]
            yield [batch_input1_samples, batch_input2_samples], batch_labels


def load_csv(filename:os.path) -> pd.DataFrame:
    '''
    load csv file
    :param filename: csv file path
    :return: a dataframe
    '''
    with open(filename) as file:
        data = pd.read_csv(file)
    return data


def batch_min_max_normalization(data: list) -> list:
    '''
    This is to do global normalization with min_max method
    :param data: a list
    :return: a list of normalized data
    '''
    data = np.array(data)

    # Calculate the global minimum and maximum values
    min_val = np.min(data)
    max_val = np.max(data)

    # Perform min-max normalization
    normalized_data = (data - min_val) / (max_val - min_val)

    return list(normalized_data)


def fixed_min_max_normalization(data: list,min = -6.021, max = 147.287) -> list:
    '''
    This is to do normalization with min_max method given fixed ceiling and floor
    :param max: max reference level
    :param min: min reference level
    :param data: data
    :return: normalized data
    '''
    data = np.array(data)

    # Perform min-max normalization
    normalized_data = (data - min) / (max - min)

    normalized_data = np.clip(normalized_data, 0, 1.0)
    return normalized_data


def split_into_subsets(dataset: pd.DataFrame, num_subsets: int, sortkey: str):
    """
    Efficiently split the dataset into a specified number of subsets while maintaining the distribution of the specified label.

    Args:
        dataset (pd.DataFrame): The input DataFrame to split.
        num_subsets (int): The number of desired subsets.
        sortkey (str): The column name used to stratify the splits.

    Returns:
        list: A list of DataFrames, each representing a subset.
    """
    # Group by the specified label
    grouped = dataset.groupby(sortkey)

    # Create a list to hold the subsets
    subsets = []
    all_samples = []

    # Sample data for each label group
    for label, group in grouped:
        n_samples_per_subset = len(group) // num_subsets

        # If not enough samples for each subset, adjust
        if n_samples_per_subset == 0:
            raise ValueError(f"Not enough samples for label '{label}' to create {num_subsets} subsets.")

        # Sample the data and append to all_samples
        sampled = np.array_split(group.sample(n=len(group), random_state=42), num_subsets)
        all_samples.append(sampled)

    # Combine samples into subsets
    for i in range(num_subsets):
        subset = pd.concat([samples[i] for samples in all_samples])
        subset = subset.sample(frac=1, random_state=i)
        subsets.append(subset)
    return subsets