import cv2
import os
import numpy as np
import random
from config import data_directory

from train_scaled_model import train_scaled_model

# returns grayscale training faces and grayscale training non
def get_training_sets(scale):
    # Split the training_nonfaces into 100x100 windows and put them in training/training_nonfaces folder
    nonfaces_directory = data_directory + "/training_nonfaces"
    faces_directory = data_directory + "/training_faces"
    face_files = os.listdir(faces_directory)
    nonface_files = os.listdir(nonfaces_directory)
    training_faces_grayscale = []
    training_nonfaces_grayscale = []
    nonface_num = 0
    nonface_example_amount = 3510 # approximate, only exact if len(nonface_files) evenly divides it
    nonface_examples_per_img = nonface_example_amount // len(nonface_files)

    #for each nonface get random windows
    for nonface_file in nonface_files:
        nonface = cv2.imread(nonfaces_directory + "/" + nonface_file)
        nonface_gray = cv2.imread(nonfaces_directory + "/" + nonface_file, cv2.IMREAD_GRAYSCALE)
        nonface_len = nonface.shape[0]
        nonface_wid = nonface.shape[1]
        window_size = int(100 * scale)
        local_nonface_count = 0
        
        # while max number for this image hasnt been reached and global max hasn't been reached make windows
        while (local_nonface_count < nonface_examples_per_img and nonface_num < nonface_example_amount):
            randRow = random.randint(0, nonface_len - window_size)
            randCol = random.randint(0, nonface_wid - window_size)
            training_nonfaces_grayscale.append(np.copy(nonface_gray[randRow:randRow+window_size, randCol:randCol+window_size]))
            local_nonface_count += 1
            nonface_num += 1

    # get faces and resize based on scale
    for face_file in face_files:
        face = cv2.imread(faces_directory + "/" + face_file, cv2.IMREAD_GRAYSCALE)
        resized_face = cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        training_faces_grayscale.append(resized_face)

    
    return training_faces_grayscale, training_nonfaces_grayscale

#get desired scale training images from get_training_sets
#three different models are trained for each scale


f_25px, nf_25px = get_training_sets(.25) #returns 25x25px images for training
print("25 px training sets recieved!")
train_scaled_model(f_25px,nf_25px , 25)

f_50px, nf_50px = get_training_sets(.5)#returns 50x50px images for training
print("50 px training sets recieved!")
train_scaled_model(f_50px, nf_50px,50)

f_100px, nf_100px = get_training_sets(1)#returns 100x100px images for training
print("100 px training sets recieved!")
train_scaled_model(f_100px, nf_100px, 100)


