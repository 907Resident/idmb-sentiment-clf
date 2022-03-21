# %% Utility Module of crisp-nlp

"""
The utility module allows the user to conduct logistical and tedious steps seamlessly with repeatable functions.
"""

# %% Import necessary modules

from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np 

import time
import os

# %% Define load imdb data

def load_imdb_train_data(path_to_data, random_state=0):

    # Create path to train folder
    path_to_train = os.path.join(path_to_data, "train")

    # Get training files
    train_files = os.listdir(path_to_train)
    # idx - neg
    neg_idx = train_files.index("neg")
    path_to_train_neg = os.path.join(path_to_train, train_files[neg_idx])
    # idx - pos
    pos_idx = train_files.index("pos")
    path_to_train_pos = os.path.join(path_to_train, train_files[pos_idx])

    # Import training data
    train_data = pd.DataFrame([])
    print("... acquiring data from folder and creating dataframe ...")
    for negative, positive in tqdm(
        zip(
            os.listdir(path_to_train_neg), os.listdir(path_to_train_pos)
            )
        ):

        # Open and read the negative instance of text into a variable
        with open(os.path.join(path_to_train_neg, negative), "r") as reader:
            negative_plain_text = reader.read()
        # Open and read the positive instance of text into a variable
        with open(os.path.join(path_to_train_pos, positive), "r") as reader:
            positive_plain_text = reader.read()

        # Transfer the text for the negative review from the training folder into a dataframe
        df_neg = pd.DataFrame({"text": [negative_plain_text], "sentiment": [0]})
        # Transfer the the positive review from the training folder into a dataframe
        df_pos = pd.DataFrame({"text": [positive_plain_text], "sentiment": [1]})
        # Concatenate the negative and positive reivews on to the training dataframe
        train_data = pd.concat([train_data, df_neg, df_pos])

    # Shuffle rows to eliminate neg-pos alteration
    train_data = (
        train_data
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return train_data
# %%
