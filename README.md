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
