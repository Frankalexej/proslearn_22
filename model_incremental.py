"""
Here we put tools for incremental training. 
"""
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import random
import math
from collections import OrderedDict, deque, Counter
from torch.utils.data import DataLoader

# Import custom modules
from PretermDataLoader import DataLoader as PretermDataLoader
from model_configs import TrainingConfigs


class ConstructDatasetGroup: 
    """
    This is to construc a group of small datasets for incremental learning. 
    It should achieve two things: 
    1. use Ming's dataloader to sample and load the corresponding dataset. 
    2. save the small datasets' meta files and the combined .npy files for later training use. 

    NOTE: This function does not check the existence of meta files. Thus it should be checked in training code. 
    """
    def __init__(self, src_path, all_meta_filename, target_dir, target_name, pre_select={}): 
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
        all_meta['index'] = all_meta.index  # add index column for later use. USED? 
        # filter the metadata before any data is noted down. 
        # all_meta_filtered = ConstructDatasetGroup.filter_dataframe(all_meta, pre_select)
        # all_meta_filtered = all_meta
        # raw_data_loader.update_metadata(all_meta_filtered)  # update the metadata in the dataloader.
        all_size = len(all_meta)    # size of the whole dataset.

        # self.all_meta = all_meta_filtered
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

        # selcol_all_meta = self.all_meta[select_column]  # select the columns
        selcol_all_meta = self.raw_data_loader.metadata[select_column]  # select the columns, but always follow what is used when selecting mels. 

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

    def append_construct(self, num_dataset=10, data_type="mel", append_pass="high"): 
        for i in range(num_dataset): 
            selected_meta = pd.read_csv(os.path.join(self.target_dir, f"{self.target_name}-{i}.csv"))
            selected_indexes = selected_meta["index"].tolist()
            data_append, _ = self.raw_data_loader.load_data(f"{append_pass}pass_{data_type}", len(selected_indexes), selected_indexes)
            print(f"Dataset {self.target_name}-{i}-{append_pass} loaded.")
            np.save(os.path.join(self.target_dir, f"{self.target_name}-{append_pass}-{i}.npy"), np.array(data_append))



class SubsetCache: 
    """
    A cache for storing subsets of a dataset in memory (i.e. our group of datasets).
    Utilizes a Least Recently Used (LRU)
    eviction policy to manage memory efficiently by limiting the number of subsets stored at a time.
    """
    def __init__(self, max_cache_size, dataset_class, shuffle=True): 
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
        self.batch_size = TrainingConfigs.BATCH_SIZE
        self.loader_worker = TrainingConfigs.LOADER_WORKER

    def get_subset(self, subset_id, subset_meta_path, subset_data_path, subset_mapper=None, shuffle=True): 
        """
        Retrieves a dataset subset from the cache or loads it from disk if not in the cache.

        Args:
            subset_id (any): A unique identifier for the subset (e.g., an index or filename).
            subset_meta_path (str): Path to the metadata file for the subset.
            subset_data_path (str): Path to the data file for the subset.
            subset_mapper (optional): An optional mapper object for custom data handling; default is None.

        Returns:
            dataloader (DataLoader): A DataLoader object for the subset.
                                     
        """
        # Check if subset is in cache
        if subset_id in self.cache: 
            # Move accessed subset to end (most recently used)
            self.cache.move_to_end(subset_id)
            return self.cache[subset_id]
        
        # Load from disk if not in cache
        dataset = self.dataset_class(subset_meta_path, subset_data_path, subset_mapper)
        dataloder = DataLoader(dataset, batch_size=self.batch_size, 
                               shuffle=shuffle, num_workers=self.loader_worker, 
                               drop_last=True)
        
        # Add to cache
        if len(self.cache) >= self.max_cache_size:
            # Remove the least recently used (LRU) item
            self.cache.popitem(last=False)
        
        # Store the new subset in cache and mark it as most recently used
        self.cache[subset_id] = dataloder
        return dataloder


class LearningPathPlanner:
    """
    A planner for generating a learning path from a pool of dataset IDs, where revisit probability exponentially decreases 
    with each visit.
    """

    def __init__(self, dataset_ids, total_epochs, p1=0.5, decay_rate=0.5):
        """
        Initializes the LearningPathPlanner.

        Args:
            dataset_ids (list): List of dataset IDs representing the pool of available datasets.
            total_epochs (int): The total number of epochs for which to generate a learning path.
            p1 (float): Probability of selecting a new dataset in each epoch (between 0 and 1).
            decay_rate (float): The rate at which revisit probability decreases exponentially with each visit.
        """
        self.dataset_ids = dataset_ids  # Pool of dataset IDs
        self.total_epochs = total_epochs
        self.p1 = p1
        self.decay_rate = decay_rate
        self.new_datasets = set(dataset_ids)  # Datasets not yet seen
        self.old_datasets = deque()  # Queue to track datasets that have been used
        self.visit_count = Counter()  # Counter to track the number of visits for each dataset

    def get_exponential_probability(self, visits):
        """
        Calculates the probability weight for a dataset based on its visit count using exponential decay.

        Args:
            visits (int): The number of times the dataset has been visited.

        Returns:
            float: The probability weight for the dataset.
        """
        return math.exp(-self.decay_rate * visits)

    def get_next_dataset(self):
        """
        Determines the next dataset ID to use based on the learning path logic, with exponential decay in revisit probability.

        Returns:
            int: The ID of the next dataset to use for training.
        """
        # Decide whether to select a new or an old dataset based on probability p1
        if self.new_datasets and random.random() < self.p1:
            # Choose a new dataset
            next_dataset = self.new_datasets.pop()
            self.old_datasets.append(next_dataset)  # Move to old datasets
            self.visit_count[next_dataset] = 0  # Initialize visit count

        else:
            # Choose an old dataset with probability exponentially decreasing by visit count
            if self.old_datasets:
                # Calculate weights for old datasets based on exponential decay
                weights = [self.get_exponential_probability(self.visit_count[ds]) for ds in self.old_datasets]
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]

                # Randomly choose an old dataset based on calculated probabilities
                next_dataset = random.choices(list(self.old_datasets), weights=probabilities, k=1)[0]
                self.visit_count[next_dataset] += 1  # Increment visit count

            else:
                # Fallback to a new dataset if no old datasets are available
                next_dataset = self.new_datasets.pop()
                self.old_datasets.append(next_dataset)
                self.visit_count[next_dataset] = 0

        return next_dataset

    def generate_learning_path(self):
        """
        Generates a full learning path for the specified total number of epochs.

        Returns:
            list: A list of dataset IDs representing the planned learning path.
        """
        learning_path = []
        for _ in range(self.total_epochs):
            next_dataset = self.get_next_dataset()
            learning_path.append(next_dataset)
        return learning_path
    
class LearningPathPickup: 
    def __init__(self, learning_path, dataset_pool, on_same="random", on_end="random"): 
        self.learning_path = learning_path
        self.dataset_pool = dataset_pool
        self.on_same = on_same
        self.on_end = on_end

    def get_this_and_next(self, current_learning_position): 
        # This ensures this_id != next_id, which is the most important. 
        this_id = self.learning_path[current_learning_position]
        if current_learning_position == len(self.learning_path) - 1: 
            if self.on_end == "random": 
                return this_id, random.choice(list(set(self.dataset_pool)-set([this_id])))    # pick a random dataset not the current one
            else: 
                raise ValueError("Invalid on_end value.") 
        else: 
            next_id = self.learning_path[current_learning_position + 1]
            if next_id == this_id: 
                # we have the same next dataset. 
                if self.on_same == "random": 
                    return this_id, random.choice(list(set(self.dataset_pool)-set([this_id])))
                else: 
                    raise ValueError("Invalid on_same value.")
            else: 
                return this_id, next_id



class PoolMessanger: 
    """
    Holds the pool of datasets and gets the correct names to load the datasets. 
    This is because LearningPathPlanner only operates on the dataset IDs. 
    While loading datasets needs filenames of data and meta. 
    """
    META_SUFFIX = "csv"
    DATA_SUFFIX = "npy"
    def __init__(self, num_dataset, non_full_type="low", full_type="full", read_dir=None): 
        """
        Args:
            num_dataset (int): The number of datasets in the pool.
            non_full_type (str): type of non-full data. low, high, full. 
            full_type (str): type of full data. low, high, full. 

        Returns:
            None
        """
        self.pool = list(range(num_dataset))
        self.non_full_type = non_full_type
        self.full_type = full_type
        self.read_dir = read_dir
        self.full_on = False # marking whether the dataset is filtered or full. 
    
    def get_pool(self): 
        return self.pool
    
    def turn_on_full(self): 
        self.full_on = True

    def turn_off_full(self):
        self.full_on = False
    
    def get_loading_params(self, dataset_id, eval_type="train"): 
        if eval_type == "train" or eval_type == "valid": 
            filter_type = self.non_full_type if not self.full_on else self.full_type

            meta_path = f"{eval_type}-{dataset_id}.{self.META_SUFFIX}"
            data_path = f"{eval_type}-{filter_type}-{dataset_id}.{self.DATA_SUFFIX}"
        
        elif eval_type == "full_valid":
            meta_path = f"valid-{dataset_id}.{self.META_SUFFIX}"
            data_path = f"valid-full-{dataset_id}.{self.DATA_SUFFIX}"

        return dataset_id, os.path.join(self.read_dir, meta_path), os.path.join(self.read_dir, data_path)