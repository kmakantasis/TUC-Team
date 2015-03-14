# -*- coding: utf-8 -*-
import numpy as np

def ImageDatasetCreation(csv_name='data/trainLabels.csv', labels_idx=[0,4], number_of_data=[100, 100]):
    all_data_names = np.genfromtxt(csv_name,usecols=(0),delimiter=',',dtype=None, skip_header=1)
    all_data_labels = np.genfromtxt(csv_name,usecols=(1),delimiter=',',dtype=float, skip_header=1)

    all_data_labeled_0 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==0]
    all_data_labeled_1 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==1]
    all_data_labeled_2 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==2]
    all_data_labeled_3 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==3]
    all_data_labeled_4 = [all_data_names[i] for i in range(len(all_data_labels)) if all_data_labels[i]==4]
    
    all_data_labeled = [all_data_labeled_0, all_data_labeled_1, all_data_labeled_2, all_data_labeled_3, all_data_labeled_4]
    
    data_to_use_names = []
    for i in range(len(labels_idx)):
        assert len(all_data_labeled[labels_idx[i]]) > number_of_data[i]
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
    
    
#names, labels = ImageDatasetCreation()


