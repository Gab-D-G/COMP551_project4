# COMP551_project4

Authors: Vincent Bazinet, Gabriel Desrosiers-Grégoire, Mila Urose

Allen_brain_query.py:
Downloading the ISH data with expression maps

generate_GO_data.py:
Get the dictionary of GO terms from the downloaded .csv files from AmiGO

SIFT_features.m:
Script to extract SIFT features from images (which must all be in
the same folder). This script relies on the vlfeat library, available on 
http://www.vlfeat.org/install-matlab.html.

SIFT_features.py:
Code that extracts the SIFT figures from a set of images. These images must all be in the same folder.
This code necessitates the use of an older version of the contrib version of the openCV package (because SIFT is a patented algorithm...). 
TRY DOING:
pip uninstall opencv-python 

[If previously installed]. Then both

pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16

prep_data.ipynb: loading of the images and removal of the bottom 20% with the lowest number of non-zero pixels

PCA.ipynb: notebook for generating the PCA features

autoencoder.ipynb: notebook for training the autoencoders and generating its features.

random_search.py: script which runs the random search on one GO term with the autoencoder features, given a GO term index to select from the GO term array.

GO_pred.py: Script which runs the prediction for a subset of the GO terms from the autoencoder features, given a chunk of index to select from the GO array.

SIFT_GO_pred.py: Same as GO_pred.py, but for the SIFT features

analysis.ipynb: notebook which regroups the final analysis for the baseline comparisons.
