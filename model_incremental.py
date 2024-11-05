"""
Here we put tools for incremental training. 
"""
# Import necessary libraries
import os
import numpy as np
import pandas as pd
from collections import OrderedDict, deque, Counter

# Import custom modules
from PretermDataLoader import DataLoader as PretermDataLoader


class ConstructDatasetGroup: 
    """
    This is to construc a group of small datasets for incremental learning. 
    It should achieve two things: 
    1. use Ming's dataloader to sample and load the corresponding dataset. 
    2. save the small datasets' meta files and the combined .npy files for later training use. 

    NOTE: This function does not check the existence of meta files. Thus it should be checked in training code. 
    """
    def __init__(self, src_path, all_meta_filename, target_dir, target_name): 
        """
        Initializes the ConstructDatasetGroup object.

        Args:
            src_path (str): The path to the directory containing the raw dataset files.
            all_meta_filename (str): The filename of metadata of the whole dataset. 
            target_dir (str): The directory path where the small datasets and metadata will be saved.
            target_name (str): The base name for the saved dataset and metadata files.

        Returns:
            None
        """
        raw_data_loader = PretermDataLoader(src_path)
        all_meta = raw_data_loader.get_metadata(all_meta_filename)  # get metadata, NOTE: has to be called before other use. 
        all_meta['index'] = all_meta.index
        all_size = len(all_meta)    # size of the whole dataset.

        self.all_meta = all_meta
        self.all_size = all_size
        self.raw_data_loader = raw_data_loader
        self.target_dir = target_dir
        self.target_name = target_name

    def construct(self, num_dataset=10, size_dataset=1600, absolute_size=True, data_type="mel", select_column=["index"]): 
        """
        Constructs and saves a specified number of small datasets for incremental learning.

        Args:
            num_dataset (int): The number of small datasets to create.
            size_dataset (int): The size of each small dataset, either as an absolute number of samples or as a percentage.
            absolute_size (bool): If True, interprets size_dataset as the absolute number of samples; if False, interprets it as a percentage of the total dataset size.
            data_type (str): The type of data to load (e.g., "mel").
            select_column (list of str): List of columns to include in the metadata for each dataset.

        Returns:
            None
        """
        if not absolute_size: 
            # percentage to absolute
            size_sample = self.all_size * size_dataset
        else: 
            size_sample = size_dataset

        selcol_all_meta = self.all_meta[select_column]  # select the columns

        for i in range(num_dataset): 
            data, indexes = self.raw_data_loader.load_data(data_type, size_sample)   # load the data (full)
            print(f"Dataset {self.target_name}-{i}-full loaded.")
            data_lowpass, _ = self.raw_data_loader.load_data(f"lowpass_{data_type}", size_sample, indexes)   # load the data (low)
            print(f"Dataset {self.target_name}-{i}-low loaded.")
            data_highpass, _ = self.raw_data_loader.load_data(f"highpass_{data_type}", size_sample, indexes)   # load the data (high)
            print(f"Dataset {self.target_name}-{i}-high loaded.")
            selcol_meta = selcol_all_meta.iloc[indexes]  # select the corresponding metadata
            # save the metadata
            selcol_meta.to_csv(os.path.join(self.target_dir, f"{self.target_name}-{i}.csv"), index=False)
            # transform data (list of np array) into nparray
            np.save(os.path.join(self.target_dir, f"{self.target_name}-full-{i}.npy"), np.array(data))
            np.save(os.path.join(self.target_dir, f"{self.target_name}-low-{i}.npy"), np.array(data_lowpass))
            np.save(os.path.join(self.target_dir, f"{self.target_name}-high-{i}.npy"), np.array(data_highpass))


class SubsetCache: 
    """
    A cache for storing subsets of a dataset in memory (i.e. our group of datasets).
    Utilizes a Least Recently Used (LRU)
    eviction policy to manage memory efficiently by limiting the number of subsets stored at a time.
    """
    def __init__(self, max_cache_size, dataset_class): 
        """
        Initializes the SubsetCache object.

        Args:
            max_cache_size (int): The maximum number of subsets to keep in memory simultaneously.
            dataset_class (type): The class of the dataset to be loaded and stored in the cache. Each subset is an 
                                  instance of this class.
        """
        self.cache = OrderedDict()  # Cache for subsets (with LRU eviction policy)
        self.max_cache_size = max_cache_size
        self.dataset_class = dataset_class  # The class of the dataset to be stored and loaded in the cache.

    def get_subset(self, subset_id, subset_meta_path, subset_data_path, subset_mapper=None): 
        """
        Retrieves a dataset subset from the cache or loads it from disk if not in the cache.

        Args:
            subset_id (any): A unique identifier for the subset (e.g., an index or filename).
            subset_meta_path (str): Path to the metadata file for the subset.
            subset_data_path (str): Path to the data file for the subset.
            subset_mapper (optional): An optional mapper object for custom data handling; default is None.

        Returns:
            dataset (dataset_class): An instance of the dataset class for the specified subset, either loaded from
                                     cache or newly created and added to the cache.
        """
        # Check if subset is in cache
        if subset_id in self.cache: 
            # Move accessed subset to end (most recently used)
            self.cache.move_to_end(subset_id)
            return self.cache[subset_id]
        
        # Load from disk if not in cache
        dataset = self.dataset_class(subset_meta_path, subset_data_path, subset_mapper)
        
        # Add to cache
        if len(self.cache) >= self.max_cache_size:
            # Remove the least recently used (LRU) item
            self.cache.popitem(last=False)
        
        # Store the new subset in cache and mark it as most recently used
        self.cache[subset_id] = dataset
        return dataset


class LearningPathPlanner:
    """
    A planner for generating a learning path from a pool of dataset IDs, controlling how frequently to revisit 
    previously seen datasets.
    """

    def __init__(self, dataset_ids, revisit_frequency=3):
        """
        Initializes the LearningPathPlanner.

        Args:
            dataset_ids (list): List of dataset IDs representing the pool of available datasets.
            revisit_frequency (int): Controls how often to revisit previously seen datasets.
                                     A lower number means revisits occur sooner.
        """
        self.dataset_ids = dataset_ids  # Pool of available dataset IDs
        self.revisit_frequency = revisit_frequency
        self.seen_datasets = deque(maxlen=revisit_frequency)  # Queue to track recently seen datasets
        self.seen_count = Counter()  # Counter to keep track of how often each dataset has been seen

    def get_next_dataset(self):
        """
        Determines the next dataset ID to use based on the learning path.

        Returns:
            int: The ID of the next dataset to use for training.
        """
        # Filter to prioritize unseen datasets or datasets seen least frequently
        candidates = [ds for ds in self.dataset_ids if ds not in self.seen_datasets]
        if not candidates:
            # If all datasets have been seen recently, allow revisits based on least seen count
            candidates = self.dataset_ids

        # Select the next dataset with priority to those seen least often
        next_dataset = min(candidates, key=lambda ds: self.seen_count[ds])

        # Update seen records
        self.seen_datasets.append(next_dataset)
        self.seen_count[next_dataset] += 1

        return next_dataset

    def generate_learning_path(self, num_epochs):
        """
        Generates a full learning path for a specified number of epochs.

        Args:
            num_epochs (int): The total number of epochs for which to generate a learning path.

        Returns:
            list: A list of dataset IDs representing the planned learning path.
        """
        learning_path = []
        for _ in range(num_epochs):
            next_dataset = self.get_next_dataset()
            learning_path.append(next_dataset)
        return learning_path