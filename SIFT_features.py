# -*- coding: utf-8 -*-
"""
Created on Tue Apr 9 19:48:41 2019

Code that extracts the SIFT figures from a set of images.
These images must all be in the same folder.

This code necessitates the use of an older version of the contrib version of 
the openCV package (because SIFT is a patented algorithm...). 

TRY DOING:
    
pip uninstall opencv-python 

[If previously installed]. Then both

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16

@author: Vincent Bazinet
"""
import cv2
import numpy as np
from os import listdir
import os.path as op
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

'''
CONSTANT PARAMETERS
'''
extract_nb = 15945            #nb of images that we want to compute 
path = './my_new_folder_2/'   #path to the folder where the images are
'''
OUR FUNCTIONS
'''
def extract_BoW(images):
    '''
    Functions that extracts the SIFT descriptors from the images,
    at 4 different resolutions, cllusters these descriptors into
    500 clusters, for each resolution and returns the centroids of these
    clusters
    '''
    
    kmeans = KMeans(n_clusters=500)
    
    #Array that stores the concatenated descriptors (at 4 different resolutions)
    sift_all_1 = np.array([], dtype=np.int64).reshape(0,128)
    sift_all_2 = np.array([], dtype=np.int64).reshape(0,128)
    sift_all_4 = np.array([], dtype=np.int64).reshape(0,128)
    sift_all_8 = np.array([], dtype=np.int64).reshape(0,128)
    
    #For each images, get the descriptors (at 4 different resolutions)
    imageNb = 0        
    for i in range(extract_nb):
        
        image = images[i]
        print(imageNb)
        
        file = './my_new_folder_2/'+image
        img = cv2.imread(file)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(4):
        
            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray_img,None)
            
            #get descriptors of the images
            kp,des = sift.compute(gray_img, kp)   
            
            if i==0:
                if des is not None:
                    sift_all_1 = np.vstack([sift_all_1, des])
            
            if i==1:
                if des is not None:
                    sift_all_2 = np.vstack([sift_all_2, des])
            
            if i==2:
                if des is not None:
                    sift_all_4 = np.vstack([sift_all_4, des])
            
            if i==3:
                if des is not None:
                    sift_all_8 = np.vstack([sift_all_8, des])
            
            scale_percent = 50
            width = int(gray_img.shape[1] * scale_percent / 100)
            height = int(gray_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            gray_img = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)
        
        imageNb +=1
    
    #Identify the centroids of these clusters
    dict_1 = kmeans.fit(sift_all_1).cluster_centers_
    dict_2 = kmeans.fit(sift_all_2).cluster_centers_
    dict_4 = kmeans.fit(sift_all_4).cluster_centers_
    dict_8 = kmeans.fit(sift_all_8).cluster_centers_
    
    return dict_1, dict_2, dict_4, dict_8


'''
MAIN
'''


images = [f for f in listdir(path)]

#Get the dictionaries
dict_1, dict_2, dict_4, dict_8 = extract_BoW(images)


knearest1 = NearestNeighbors(n_neighbors=1)
knearest2 = NearestNeighbors(n_neighbors=1)
knearest4 = NearestNeighbors(n_neighbors=1)
knearest8 = NearestNeighbors(n_neighbors=1)

knearest1.fit(dict_1)
knearest2.fit(dict_2)
knearest4.fit(dict_4)
knearest8.fit(dict_8)

genes_features = np.zeros((extract_nb, 2004))
imageNb = 0

#For each image, get the descriptors, and map them to the nearest neighbors
#amongst the centroids
for i in range(extract_nb):
    
    image = images[i]
    print(imageNb)
    
    file = './my_new_folder_2/'+image
    img = cv2.imread(file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(4):
    
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray_img,None)
        kp,des = sift.compute(gray_img, kp)   
            
        if i==0:
            if des is not None:
                for i in range(len(des)):
                    nearest = knearest1.kneighbors(des[i,:].reshape(1,128), 1, return_distance=False)
                    genes_features[imageNb,nearest] +=1
            else:
                genes_features[imageNb,2000] += 1
        if i==1:
            if des is not None:
                for i in range(len(des)):
                    nearest = knearest2.kneighbors(des[i,:].reshape(1,128), 1, return_distance=False)
                    genes_features[imageNb,nearest+500] += 1
            else:
                genes_features[imageNb,2001] += 1
        if i==2:
            if des is not None:
                for i in range(len(des)):
                    nearest = knearest4.kneighbors(des[i,:].reshape(1,128), 1, return_distance=False)
                    genes_features[imageNb,nearest+1000] += 1
            else:
                genes_features[imageNb,2002] += 1
        if i==3:
            if des is not None:
                for i in range(len(des)):
                    nearest = knearest8.kneighbors(des[i,:].reshape(1,128), 1, return_distance=False)
                    genes_features[imageNb,nearest+1500] += 1
            else:
                genes_features[imageNb,2003] += 1
            
        scale_percent = 50
        width = int(gray_img.shape[1] * scale_percent / 100)
        height = int(gray_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        gray_img = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA) 
        
    imageNb += 1
