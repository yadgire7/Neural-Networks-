'''
Author: Saurabh Arun Yadgire
Code referred from: patrickloeber
'''

# import required libraries
import io
import os
import unicodedata
import string
import glob
import torch
import random
import re

# string of all letters
ALL_LETTERS = string.ascii_letters + " .,;'"
NUM_LETTERS = len(ALL_LETTERS)

# remove unicode characters from string(non-alphabetical characters)
def remove_unicode(text):
    return re.sub('[^a-zA-Z\s]', '', text)

# load data from file
def load_data(path):
    category_words_dict = {}
    labels = []
    def get_files(path):
        files = glob.iglob(f"{path}*.txt")
        return files
    
    def get_class_labels_from_files(files):
        for file in files:
            label = file.split("\\")[-1].split(".")[0]
            labels.append(label)
        return labels
    
    def read_lines(path, filename):
        lines = io.open(os.path.join(path, filename), encoding='utf-8').read().strip().split('\n')
        return [remove_unicode(line) for line in lines]

    files = get_files(path)
    for file in files:
        label = file.split("\\")[-1].split(".")[0]
        labels.append(label)
        filename = file.split("\\")[-1]
        lines = read_lines(path, filename)
        category_words_dict[label] = lines 

    return category_words_dict, labels


'''
create one hot encoding vector for each word from each category
'''

def create_ohe(letter):
    return ALL_LETTERS.find(letter)

def create_letter_tensor(letter):
    tensor = torch.zeros(1, NUM_LETTERS)
    tensor[0][create_ohe(letter)] = 1
    return tensor

def create_word_tensor(word):
    tensor = torch.zeros(len(word), 1, NUM_LETTERS)
    for i, letter in enumerate(word):
        tensor[i][0][create_ohe(letter)] = 1
    return tensor

def random_training_example(category_words_dict, labels):
    def random_choice(a):
        idx = random.randint(0, len(a) - 1)
        return a[idx]
    
    category = random_choice(labels)

    word = random_choice(category_words_dict[category])
    category_tensor = torch.tensor([labels.index(category)], dtype=torch.long)
    word_tensor = create_word_tensor(word)
    return category, word, category_tensor, word_tensor
           

