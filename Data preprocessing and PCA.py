import cv2
import os
import torch
import numpy as np


train_img_dir = "C:/Users/Fried/Desktop/SYDE 660A/Codes and Data/data/train" #path of training data

f = open(train_img_dir+"/train_split_v3.txt", "r")
lines = f.readlines()
lines = [line.rstrip("\n").split(" ")[1:3] for line in lines]
lines = np.array(lines)
f.close()

test_img_dir = "C:/Users/Fried/Desktop/SYDE 660A/Codes and Data/data/test" #path of testing data

f = open(train_img_dir+"/test_split_v3.txt", "r")
lines = f.readlines()
lines = [line.rstrip("\n").split(" ")[1:3] for line in lines]
lines = np.array(lines)
f.close()


file_names = lines[:,0]
labels = lines[:,1]

all_files = os.listdir(train_img_dir)

label_files = ["train_split_v3.txt","test_split_v3.txt"]
image_files = [file for file in all_files if file not in label_files]


import collections
print([item for item, count in collections.Counter(file_names).items() if count > 1])
repeated_index = np.argwhere(file_names == '000001-27.jpg')


# Delete the repeated image
file_names = np.delete(file_names,34)
labels = np.delete(labels,34)


y_train = []

for label in labels:
    if label == 'COVID-19':
        y = 0
    if label  == 'pneumonia': 
        y = 1
    if label == 'normal':
        y = 2
    y_train.append(y)


new_label_file = np.column_stack((file_names,y_train))
np.savetxt('train_split_v3_corrected.txt',new_label_file,fmt='%s')


dim = (64,64) # Resizing the images

image_list = []
for file_name in file_names: 
    image = cv2.imread(test_img_dir+"/"+file_name,0)
    image = cv2.resize(image,dim)
    image_list.append(image)


image_list = [image.ravel() for image in image_list]
image_array = np.array(image_list)



def PCA_POV(X,percentage=0.05):
    """
   PCA based on % of variance explained
   X: image array
   percentage: 1 - % of variance explained
   
    """
    
    mean = np.mean(X,axis=0)
    var = np.var(X,axis=0)
    cov = np.dot((X-mean).T,(X-mean))/X.shape[0]
    
    u,s,vh = np.linalg.svd(cov)

    total = np.sum(s)
        
    i = 0
    while i in range(len(s)):
        f = np.sum(s[0:i])
        percent = 1-f/np.sum(s)
        i += 1
        if percent <= percentage:
            break
                
    P_PCA = u[:,:i]
    print(str(i)+' features were selected for POV of '+str(1-percentage))
        
    return { 'X':X, 'number of features':i, 'components':u, 'P': P_PCA, 'mean': mean, 'variance': var, 
            'POV':1-percent }


def PCA_k(X,k=10):
    """
   PCA based on number of desired features
   X: image array
   k: number of features to keep
   
    """
    
    mean = np.mean(X,axis=0)
    var = np.var(X,axis=0)
    cov = np.dot((X-mean).T,(X-mean))/X.shape[0]
    
    u,s,vh = np.linalg.svd(cov)

    total = np.sum(s)
    P_PCA = u[:,:k]
    
    percent = 1-np.sum(s[0:k])/total
    print('POV is')
    print(1-percent)
        
    return { 'X':X, 'number of features':k, 'components':u, 'P':P_PCA, 'mean': mean, 'variance': var, 
            'POV':1-percent }



dict_ = PCA_POV(image_array,percentage=0.15) # apply PCA on the resized images
dict_['P'].shape # Check output shape
image_array_transformed = np.dot(image_array,dict_['P']) 


PCA_components = open("PCA_data_all_v4.pkl", "wb") # save the principle components
torch.save(dict_, PCA_components)
PCA_components.close()

PCA_components = open("PCA_data_all_v4.pkl", "rb") 
output = torch.load(PCA_components)


image_with_label_PCA = np.column_stack((image_array_transformed, y_train)) # save the transformed images with labels
np.savetxt('image_with_label_POV85.txt',image_with_label_PCA,fmt='%s')

