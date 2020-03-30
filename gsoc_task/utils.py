import os
from shutil import copyfile

import pandas as pd
import numpy as np


def prepare_data(input_dir  , train_ratio=0.7):
    '''
    Create CSV file information about path of the images directory and 
    Binary Label (sub-structure or no sub-structure)

    params:
        input_dir: Input directory wher both the directories with images are present
        train_ratio: Fraction of data to be considered as training data between [0-1]

    '''
    csv_filepath = input_dir + 'lense.csv'
    if os.path.isfile(csv_filepath):
        lens_data = pd.read_csv(csv_filepath)
        print('Reading from Existing CSV File: ', csv_filepath)
    else : 
        lens_data = pd.DataFrame()

        no_sub_images = [input_dir + 'no_sub/'+ str(f) for f in os.listdir(input_dir + "/no_sub")]
        no_sub_samples = [0] *len(no_sub_images)

        sub_images = [input_dir + 'sub/'+ str(f) for f in os.listdir(input_dir + "/sub")]
        sub_samples = [1]*len(sub_images)

        lens_data['path'] = no_sub_images + sub_images
        lens_data['is_sub'] = [*no_sub_samples, *sub_samples] #Concatenate 2 lists with numbers. Only valid for Python 3.6+

        lens_data.to_csv(csv_filepath)
        print('Creating new CSV File:', csv_filepath)
    
    print("***Now creating Train-Test Split***")

    train_dir = os.path.join(input_dir + 'train/')
    test_dir = os.path.join(input_dir + 'test/')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    train_images = os.path.join(train_dir,'train.csv')
    test_images = os.path.join(test_dir,'test.csv')

    if os.path.isfile(train_images):
        ##
        print("Found Existing train.csv at :", train_images)
        print("Found Existing test.csv at :", test_images)
        train = pd.read_csv(train_images)
        test = pd.read_csv(test_images)
    else:
        #Train-Test Split
        lens_data['split'] = np.random.randn(lens_data.shape[0], 1)
        msk = np.random.rand(len(lens_data)) <= train_ratio
        train = lens_data[msk]
        test = lens_data[~msk]

        train.to_csv(train_images, columns=['path','is_sub'])
        print("Created train.csv :", train_images)
        test.to_csv(test_images, columns=['path','is_sub'])
        print("Created test.csv at :", test_images)
    
    #torch.datasets.ImageLoader() in Pytorch behaves strangely and is not able to read images
    # Creating a folder with name 1 and moving all images to it works somehow
    if not os.path.isdir(os.path.join(train_dir,"1")):
        print("creating directory with 1")
        os.mkdir(os.path.join(train_dir,"1"))

    for src in train.path:
        dst = os.path.join(os.path.join(train_dir,"1"), os.path.basename(src))
        copyfile(src,dst) 

    if not os.path.isdir(os.path.join(test_dir,"1")):
        os.mkdir(os.path.join(test_dir,"1"))

    for src in test.path:
        dst = os.path.join(os.path.join(test_dir,"1"), os.path.basename(src))
        copyfile(src,dst)

    return train,test
