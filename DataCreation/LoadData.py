# -*- coding: utf-8 -*-
import numpy as np
import os

def ImageDatasetCreation(csv_name='../data/trainLabels.csv', labels_idx=[0,4], number_of_data=[100, 100], LRB='both'):
    all_data_names = np.genfromtxt(csv_name,usecols=(0),delimiter=',',dtype=None, skip_header=1)
    all_data_labels = np.genfromtxt(csv_name,usecols=(1),delimiter=',',dtype=float, skip_header=1)
    
    ############################
    LRB_names = []
    LRB_labels = []
    if LRB != 'both':
        for i in range(all_data_names.shape[0]):
            if all_data_names[i].split('_')[1]==LRB:
                LRB_names.append(all_data_names[i])
                LRB_labels.append(all_data_labels[i])
                
        all_data_names = np.asarray(LRB_names)
        all_data_labels = np.asarray(LRB_labels)    
    ############################

    all_data_labeled_0 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==0]
    all_data_labeled_1 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==1]
    all_data_labeled_2 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==2]
    all_data_labeled_3 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==3]
    all_data_labeled_4 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==4]
    
    all_data_labeled = [all_data_labeled_0, all_data_labeled_1, all_data_labeled_2, all_data_labeled_3, all_data_labeled_4]
    
    data_to_use_names = []
    for i in range(len(labels_idx)):
        assert len(all_data_labeled[labels_idx[i]]) >= number_of_data[i]
        data_to_use_names.append(all_data_labeled[labels_idx[i]][0:number_of_data[i]])


    specific_data_to_use_names = []
    specific_data_to_use_labels = []


    for i in range(len(labels_idx)):
        specific_data_to_use_names = specific_data_to_use_names + data_to_use_names[i]
        specific_data_to_use_labels = specific_data_to_use_labels + [labels_idx[i]]*number_of_data[i]
 

    specific_data_to_use_names = np.asarray(specific_data_to_use_names)
    specific_data_to_use_labels = np.asarray(specific_data_to_use_labels, dtype=float)

    rand_perm = np.random.permutation(sum(number_of_data))
    specific_data_to_use_names = specific_data_to_use_names[rand_perm]
    specific_data_to_use_labels = specific_data_to_use_labels[rand_perm]

    return specific_data_to_use_names, specific_data_to_use_labels
    
    

def InputDataset(csv_name='../data/trainLabels.csv', input_folder='../data/input'):
    files = [f for f in os.listdir(input_folder)]

    all_data_names = np.genfromtxt(csv_name,usecols=(0),delimiter=',',dtype=None, skip_header=1)
    all_data_labels = np.genfromtxt(csv_name,usecols=(1),delimiter=',',dtype=float, skip_header=1)

    names_input = []
    labels_input = []

    for name in files:
        name_idx = np.where(all_data_names==name.split('.')[0])
        names_input.append(all_data_names[name_idx])
        labels_input.append(all_data_labels[name_idx])
        
    return np.asarray(names_input), np.asarray(labels_input)






