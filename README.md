# A-Multiple-Feature-Reuse-Network-to-Extract-Buildings-From-Remote-Sensing-Imagery
Python implementation of Convolutional Neural Network (CNN) used in paper

This repository includes functions to preprocess the input images and their respective polygons so as to create the 
input image patches and mask patches to be used for model training. The CNN used here is the 56 - convolutional - layers 
Multiple Feature Reuse Network (MFRN) implemented in the paper 'A Multiple - Feature Reuse Network to Extract Buildings from Remote 
Sensing Imagery' by Lin L., Liang J., Weng M., Zhu H. (2018)

The main differences between the implementations in the paper and the implementation in this repository is as follows:
- Sigmoid layer is used as the last layer instead of the softmax layer, in consideration of the fact that this is a binary 
  classification problem
- The dice coefficient function is used as the loss function in place of the binary cross - entropy loss function, in consideration 
  of the fact that this is a semantic segmentation problem, whereby emphasis should be placed on accuracy of target delineation
- Simple data augmentation is used to improve the rotation - invariance of target recognition by the MFRN model

Requirements:
- cv2
- glob
- json
- numpy
- keras (Tensorflow backend)
- gdal
