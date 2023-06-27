import pandas as pd
import numpy as np
import re
import os

def read_txt(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        combined_text += read_txt(file_path)
    return combined_text

# Read documents from the directory
train_directory = 'Dataset/data/Albums/'
text_data = ''
for albumName in os.listdir(train_directory):
    if(albumName) == 'SpeakNow_WorldTourLive':
        continue
    albumDir = train_directory + albumName
    training_data = read_documents_from_directory(albumDir) 
    text_data += training_data

with open("train.txt", "w", encoding = "utf8") as f:
    f.write(text_data)