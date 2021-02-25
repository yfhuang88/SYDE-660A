import h5py
import time
import cv2
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split


train_img_dir = "C:/Users/Fried/Desktop/SYDE 660A/Codes and Data/data/train" #path of training data
train_lbl_file = "train_split_v3.txt" #name of file which contains training file name and label

train_lbl = open(train_img_dir+'/'+train_lbl_file)
train_file = train_lbl.readlines()
train_lbl.close()

train_file = [l.rstrip("\n").split(" ")[1:3] for l in train_file]

train_dict = {line[0]:line[1] for line in train_file}
label_dict = {"COVID-19":0,"pneumonia":1,"normal":2}



from sklearn.model_selection import KFold # 10-Fold Cross-Validation Split
kf =  KFold(n_splits=10)
kf.get_n_splits(train_file)
print(kf)

validation_indices = []
for train_index, validation_index in kf.split(train_file):
    train_indices.append(train_index)
    validation_indices.append(validation_index)


train_file = np.array(train_file) # write the H5 file for each split
for num in range(0,10):
    fileName = 'Covid_split_{}.h5'.format(num+1)
    N_train = len(train_indices[num])
    N_validation = len(validation_indices[num])
    H = 480
    W = 480
    
    with h5py.File(fileName, "w") as out:
        out.create_dataset("X_train",(N_train,H,W),dtype='u1')
        out.create_dataset("Y_train",(N_train,1),dtype='u1')          
        out.create_dataset("X_validation",(N_validation,H,W),dtype='u1')
        out.create_dataset("Y_validation",(N_validation,1),dtype='u1')
    
    train_files = train_file[train_indices[num]][:,0]
    validation_files = train_file[validation_indices[num]][:,0]
    
    with h5py.File(fileName,'a') as out:
        s = time.time()
        i = 0
        for file_names in train_files:
            X = cv2.imread(train_img_dir+'/'+file_names,0)
            X = cv2.resize(X,(480,480))
            out['X_train'][i] = X
            
            y = label_dict[train_dict[file_names]]
            out["Y_train"][i] = y
            i += 1
            if i % 128 == 0:
                print("Training set is done at {}".format(i))
                e = time.time()
                print("Time taken for 128 images is {} s".format(e-s))
                s = time.time()
                
        j = 0
        for file_names in validation_files:
            X = cv2.imread(train_img_dir+'/'+file_names,0)
            X = cv2.resize(X,(480,480))
            out["X_validation"][j] = X

            y = label_dict[train_dict[file_names]]
            out["Y_validation"][j] = y

            j += 1
            if j % 128 == 0:
                  print("Validation set is done at {}".format(j))



class dataset_h5(torch.utils.data.Dataset):#create dataset object from H5 file
    def __init__(self, in_file, type_, transform=None):
        """
        in_file: the h5 file to open
        type_ : "train" or "validation"
        transform : optional, for data augmentation with function transform
        
        """
        super(dataset_h5, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.transform = transform
        self.type_ = type_
        
    def __getitem__(self, index):
        x = self.file['X_'+self.type_][index]
        y = self.file['Y_'+self.type_][index]
        
        # Preprocessing each image (this is an optional step for data augmentation)
        #if self.transform is not None:
            #x = self.transform(x)        
        
        return (x, y), index
    
    def __len__(self):
        return self.file['X_'+self.type_].shape[0]
