"""
Implementation of the Deep Temporal Clustering model
Dataset loading functions

@author Florent Forest (FlorentF9)
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.io as spio
import random
import h5py
import gc

def load_Benign_data(dataset_name):
    RF_obj=h5py.File(dataset_name, 'r')
        
#    inv=RF_obj["inv"][:]
#    TumorLength=RF_obj["TumorLength"][:]
#    Labels=RF_obj["label"][:]
    RF_Data=RF_obj["RF_dat"][:]
    patient_id=RF_obj["patient_id"][:]
    Core_points=RF_obj["PointsNo"][:]
    

    #benign cores for train
#    benign_patients=set(np.unique(patient_id))
#    benign_patients_train=random.sample(benign_patients,5)
    benign_patients_train=[2,4,10,13,14,24,25,27,29,39]
    
    benign_cores_tr=[]
    for i in benign_patients_train:
        benign_cores_tr.append(np.where(patient_id[:]==i))
    benign_cores_train=np.concatenate(benign_cores_tr,axis=1)

    del benign_cores_tr        
    #benign cores for test
#    benign_patients_test=benign_patients-set(benign_patients_train)
#    benign_patients_test=random.sample(benign_patients_test,5)
    benign_patients_test=[12,21,30,37,43]
    
    benign_cores_te=[]
    for i in benign_patients_test:
        benign_cores_te.append(np.where(patient_id[:]==i))
    benign_cores_test=np.concatenate(benign_cores_te,axis=1)
      
    del benign_cores_te
    data_tr_Benign=[]
    
    #Build train data
    for i in benign_cores_train[0]:
        start=sum(Core_points[0:i])
        stop=start+Core_points[i]
        data_tr_Benign.append(RF_Data[start:stop,:])
    data_train_Benign=np.concatenate(data_tr_Benign,axis=0)   
    
    
    #Build test data
#    data_test_cancer=np.squeeze(RF_Data[0,test_cancer_cid])
#    data_test_cancer=np.concatenate(data_test_cancer,axis=0)
#    Ncs=data_test_cancer.shape[0]
    
    data_test_Benign=[]
    
    #Build train data
    for i in benign_cores_test[0]:
        start=sum(Core_points[0:(i-1)])
        stop=start+Core_points[i]
        data_test_Benign.append(RF_Data[start:stop,:])
    data_test_Benign=np.concatenate(data_test_Benign,axis=0)   

    del RF_Data

    gc.collect()
    RF_obj.close()
    return data_train_Benign,data_test_Benign

def load_Cancer_data(dataset_name):
    RF_obj=h5py.File(dataset_name, 'r')
        
#    inv=RF_obj["inv"][:]
#    TumorLength=RF_obj["TumorLength"][:]
#    Labels=RF_obj["label"][:]
    RF_Data=RF_obj["RF_dat"][:]
    patient_id=RF_obj["patient_id"][:]
    Core_points=RF_obj["PointsNo"][:]
    

    #benign cores for train
#    benign_patients=set(np.unique(patient_id))
#    benign_patients_train=random.sample(benign_patients,5)
    cancer_patients_train=[3,6,8,11,16,20,22,23,28,38,44]
    
    cancer_cores_tr=[]
    for i in cancer_patients_train:
        cancer_cores_tr.append(np.where(patient_id[:]==i))
    cancer_cores_train=np.concatenate(cancer_cores_tr,axis=1)
        
    del cancer_cores_tr
    #benign cores for test
#    benign_patients_test=benign_patients-set(benign_patients_train)
#    benign_patients_test=random.sample(benign_patients_test,5)
#    cancer_patients_test=[5,7,15,18,19,41]
    
    cancer_cores_test=[]
    for i in cancer_patients_test:
        cancer_cores_test.append(np.where(patient_id[:]==i))
    cancer_cores_test=np.concatenate(cancer_cores_test,axis=1)
      
    data_tr_cancer=[]
    
    #Build train data
    for i in cancer_cores_train[0]:
        start=sum(Core_points[0:i])
        stop=start+Core_points[i]
        data_tr_cancer.append(RF_Data[start:stop,:])
    data_train_cancer=np.concatenate(data_tr_cancer,axis=0)   
    del data_tr_cancer
    
    #Build test data
#    data_test_cancer=np.squeeze(RF_Data[0,test_cancer_cid])
#    data_test_cancer=np.concatenate(data_test_cancer,axis=0)
#    Ncs=data_test_cancer.shape[0]
    
    data_test_cancer=[]
    
    #Build train data
    for i in cancer_cores_test[0]:
        start=sum(Core_points[0:(i-1)])
        stop=start+Core_points[i]
        data_test_cancer.append(RF_Data[start:stop,:])
    data_test_cancer=np.concatenate(data_test_cancer,axis=0)   

    del RF_Data

    gc.collect()
    RF_obj.close()
    return data_train_cancer,data_test_cancer

def load_data(dataset_name):
    RF_obj=h5py.File(dataset_name, 'r')
        
    inv=RF_obj["inv"][:]
    TumorLength=RF_obj["TumorLength"][:]
    Labels=RF_obj["label"][:]
    RF_Data=RF_obj["RF_dat"][:]
    patient_id=RF_obj["patient_id"][:]
    Core_points=RF_obj["PointsNo"][:]
    
    
    #Good Cancer
    high_cancer_indices=np.where(np.logical_and(np.logical_or(inv >= 30, TumorLength>=6), Labels==1))
    cancer_pid=patient_id[high_cancer_indices]
    
    
    #train Cancer patients=[3,6,11,22,23,28,38]
    train_cancer=[]
    train_cancer_patients=[3,6,8,11,16,20,22,23,28,38,44]
    for i in train_cancer_patients:
      train_cancer.append(np.where(cancer_pid==i))
    train_cancer=np.concatenate(train_cancer,axis=1)
    train_cancer_cid=high_cancer_indices[0][train_cancer[0]]

    #test Cancer patients=[5,7,18,19]
    test_cancer=[]
    test_cancer_patients=[5,7,15,18,19,41]
    for i in test_cancer_patients:
      test_cancer.append(np.where(cancer_pid==i))
    test_cancer=np.concatenate(test_cancer,axis=1)
    test_cancer_cid=high_cancer_indices[0][test_cancer[0]]
    
    #find benign cores from all benign patients
    cancer_cid=np.where(Labels==1)
    cancer_pid=np.unique(patient_id[cancer_cid])
    #benign cores for train
    benign_patients=set(np.unique(patient_id[:]))-set(cancer_pid )-set([9]) # 9th patient is special case
    benign_patients_train=random.sample(benign_patients,5)
    
    benign_cores_train=[]
    for i in benign_patients_train:
        benign_cores_train.append(np.where(patient_id[:]==i))
    benign_cores_train=np.concatenate(benign_cores_train,axis=1)
        
    #benign cores for test
    benign_patients_test=benign_patients-set(benign_patients_train)
    benign_patients_test=random.sample(benign_patients_test,5)
    
    benign_cores_test=[]
    for i in benign_patients_test:
        benign_cores_test.append(np.where(patient_id[:]==i))
    benign_cores_test=np.concatenate(benign_cores_test,axis=1)
      
    data_train_Benign=[]
    
    #Build train data
    for i in benign_cores_train[0]:
        start=sum(Core_points[0:i])
        stop=start+Core_points[i]
        data_train_Benign=data_train_Benign.append(RF_Data[start:stop,:])
    data_train_Benign=np.concatenate(data_train_Benign,axis=0)   
    
    
    #Build test data
#    data_test_cancer=np.squeeze(RF_Data[0,test_cancer_cid])
#    data_test_cancer=np.concatenate(data_test_cancer,axis=0)
#    Ncs=data_test_cancer.shape[0]
    
    data_test_Benign=[]
    
    #Build train data
    for i in benign_cores_test:
        start=sum(Core_points[0:(i-1)])
        stop=start+Core_points[i]
        data_test_Benign=data_test_Benign.append(RF_Data[start:stop,:])
    data_test_Benign=np.concatenate(data_test_Benign,axis=0)   

    
    return data_train_Benign,data_test_Benign

