import cv2
import glob
import json
import numpy as np
from keras import backend as K
from keras.models import Input, Model
from keras.layers.core import Dropout
from keras.layers import AveragePooling2D, BatchNormalization, concatenate, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from osgeo import gdal



def training_mask_generation(input_image_filename, input_geojson_filename):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    image = gdal.Open(input_image_filename)
    mask = np.zeros((image.RasterYSize, image.RasterXSize))
    
    ulx, xres, xskew, uly, yskew, yres = image.GetGeoTransform()                                   
    lrx = ulx + (image.RasterXSize * xres)                                                         
    lry = uly - (image.RasterYSize * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((image.RasterXSize) ** 2 / (image.RasterXSize + 1)) / (lrx - ulx)
        yf = ((image.RasterYSize) ** 2 / (image.RasterYSize + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask, position, 1)
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, image_height_size, image_width_size, mode, percentage_overlap, 
                                      buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image and its mask in the form of fixed size segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((image_array.shape[0], x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = mask_array
        
    img_list = []
    mask_list = []
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, 0]
            mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
            mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
            mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
            mask_flip_hor = cv2.flip(mask_original, 0)
            mask_flip_vert = cv2.flip(mask_original, 1)
            mask_flip_both = cv2.flip(mask_original, -1)
            mask_list.extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, mask_flip_vert, 
                              mask_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(mask_list), image_height_size, image_width_size, 1))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        mask_segment_array[index, :, :, 0] = mask_list[index]
        
    return image_segment_array, mask_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training

    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\' + 'Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Training Polygons' + '\\Train_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    
    for file in range(len(img_files)):
        img = np.transpose(gdal.Open(img_files[file]).ReadAsArray(), [1, 2, 0])
        mask = training_mask_generation(img_files[file], polygon_files[file])
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 0, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 1, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 2, 
                                                                      percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 3, 
                                                                      percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    
    return img_full_array, mask_full_array



def dice_coef(y_true, y_pred):
    """ 
    This function generates the dice coefficient for use in semantic segmentation model training. 
    
    """
    
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    coef = (2 * intersection) / (K.sum(y_true_flat) + K.sum(y_pred_flat))
    
    return coef



def dice_coef_loss(y_true, y_pred):
    """ 
    This function generates the dice coefficient loss function for use in semantic segmentation model training. 
    
    """
    
    return -dice_coef(y_true, y_pred)
    

    
def MFRN_model(img_height_size, img_width_size, n_bands, initial_num_filters, num_block_layer_filters, 
              perc_skip_filter, perc_deconv_filter, dropout_rate, l_r):
    """ 
    This function is used to replicate the 56 - convolutional - layers network architecture used in the paper 
    'A Multiple-Feature Reuse Network to Extract Buildings from Remote Sensing Imagery' by Li L., Liang J., Weng M., Zhu H. 
    (2018)
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - initial_num_filters: Number of convolutional layers to be used for the very first convolutional layer
    - num_block_layer_filters: Number of convolutional layers to be used for each layer in each dense block
    - perc_skip_filter: Percentage of feature maps to be kept for each skip connection
    - perc_deconv_filter: Percentage of feature maps to be kept for each compression transition (transposed convolution)
    - dropout_rate: Dropout rate to be used during model training
    - l_r: Learning rate to be applied for the Adam optimizer
    
    Outputs:
    - mfrn_model: Multiple Feature Reuse Network (MFRN) model to be trained using input parameters and network architecture
    
    """
    
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for dropout_rate.')
        
    if perc_skip_filter < 0 or perc_skip_filter > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc_skip_filter.')
        
    if perc_deconv_filter < 0 or perc_deconv_filter > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc_deconv_filter.')
        
    block_1_size = initial_num_filters + 2 * num_block_layer_filters
    block_2_size = block_1_size + 4 * num_block_layer_filters
    block_3_size = block_2_size + 4 * num_block_layer_filters
    block_4_size = block_3_size + 4 * num_block_layer_filters
    block_5_size = block_4_size + 4 * num_block_layer_filters
    block_6_size = block_5_size + 4 * num_block_layer_filters
    block_7_size = int(perc_skip_filter * block_5_size) + int(perc_deconv_filter * block_6_size) + 4 * num_block_layer_filters
    block_8_size = int(perc_skip_filter * block_4_size) + int(perc_deconv_filter * block_7_size) + 4 * num_block_layer_filters
    block_9_size = int(perc_skip_filter * block_3_size) + int(perc_deconv_filter * block_8_size) + 4 * num_block_layer_filters
    block_10_size = int(perc_skip_filter * block_2_size) + int(perc_deconv_filter * block_9_size) + 4 * num_block_layer_filters

    
    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    conv_initial = Conv2D(initial_num_filters, (3, 3), padding = 'same', activation = 'relu')(img_input)
    
    batch_norm_1 = BatchNormalization()(conv_initial)
    layer_1_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_1)
    conv_1_layer_1_1 = concatenate([batch_norm_1, layer_1_1])
    layer_1_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_1_layer_1_1)
    dense_block_1 = concatenate([batch_norm_1, layer_1_1, layer_1_2])
    dense_block_1 = Dropout(dropout_rate)(dense_block_1)
    
    skip_connection_1 = Conv2D(int(perc_skip_filter * block_1_size), (3, 3), 
                               padding = 'same', activation = 'relu')(dense_block_1)
    pool_1 = AveragePooling2D(pool_size = (2, 2))(dense_block_1)
    
    batch_norm_2 = BatchNormalization()(pool_1)
    layer_2_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_2)
    conv_2_layer_2_1 = concatenate([batch_norm_2, layer_2_1])
    layer_2_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_2_layer_2_1)
    conv_2_layer_2_2 = concatenate([batch_norm_2, layer_2_1, layer_2_2])
    layer_2_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_2_layer_2_2)
    conv_2_layer_2_3 = concatenate([batch_norm_2, layer_2_1, layer_2_2, layer_2_3])
    layer_2_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_2_layer_2_3)
    dense_block_2 = concatenate([batch_norm_2, layer_2_1, layer_2_2, layer_2_3, layer_2_4])
    dense_block_2 = Dropout(dropout_rate)(dense_block_2)
    
    skip_connection_2 = Conv2D(int(perc_skip_filter * block_2_size), (3, 3), 
                               padding = 'same', activation = 'relu')(dense_block_2)
    pool_2 = AveragePooling2D(pool_size = (2, 2))(dense_block_2)
    
    batch_norm_3 = BatchNormalization()(pool_2)
    layer_3_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_3)
    conv_3_layer_3_1 = concatenate([batch_norm_3, layer_3_1])
    layer_3_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_3_layer_3_1)
    conv_3_layer_3_2 = concatenate([batch_norm_3, layer_3_1, layer_3_2])
    layer_3_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_3_layer_3_2)
    conv_3_layer_3_3 = concatenate([batch_norm_3, layer_3_1, layer_3_2, layer_3_3])
    layer_3_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_3_layer_3_3)
    dense_block_3 = concatenate([batch_norm_3, layer_3_1, layer_3_2, layer_3_3, layer_3_4])
    dense_block_3 = Dropout(dropout_rate)(dense_block_3)
    
    skip_connection_3 = Conv2D(int(perc_skip_filter * block_3_size), (3, 3), 
                               padding = 'same', activation = 'relu')(dense_block_3)
    pool_3 = AveragePooling2D(pool_size = (2, 2))(dense_block_3)
    
    batch_norm_4 = BatchNormalization()(pool_3)
    layer_4_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_4)
    conv_4_layer_4_1 = concatenate([batch_norm_4, layer_4_1])
    layer_4_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_4_layer_4_1)
    conv_4_layer_4_2 = concatenate([batch_norm_4, layer_4_1, layer_4_2])
    layer_4_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_4_layer_4_2)
    conv_4_layer_4_3 = concatenate([batch_norm_4, layer_4_1, layer_4_2, layer_4_3])
    layer_4_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_4_layer_4_3)
    dense_block_4 = concatenate([batch_norm_4, layer_4_1, layer_4_2, layer_4_3, layer_4_4])
    dense_block_4 = Dropout(dropout_rate)(dense_block_4)
    
    skip_connection_4 = Conv2D(int(perc_skip_filter * block_4_size), (3, 3), 
                               padding = 'same', activation = 'relu')(dense_block_4)
    pool_4 = AveragePooling2D(pool_size = (2, 2))(dense_block_4)
    
    batch_norm_5 = BatchNormalization()(pool_4)
    layer_5_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_5)
    conv_5_layer_5_1 = concatenate([batch_norm_5, layer_5_1])
    layer_5_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_5_layer_5_1)
    conv_5_layer_5_2 = concatenate([batch_norm_5, layer_5_1, layer_5_2])
    layer_5_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_5_layer_5_2)
    conv_5_layer_5_3 = concatenate([batch_norm_5, layer_5_1, layer_5_2, layer_5_3])
    layer_5_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_5_layer_5_3)
    dense_block_5 = concatenate([batch_norm_5, layer_5_1, layer_5_2, layer_5_3, layer_5_4])
    dense_block_5 = Dropout(dropout_rate)(dense_block_5)
    
    skip_connection_5 = Conv2D(int(perc_skip_filter * block_5_size), (3, 3), 
                               padding = 'same', activation = 'relu')(dense_block_5)
    pool_5 = AveragePooling2D(pool_size = (2, 2))(dense_block_5)
    
    batch_norm_6 = BatchNormalization()(pool_5)
    layer_6_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_6)
    conv_6_layer_6_1 = concatenate([batch_norm_6, layer_6_1])
    layer_6_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_6_layer_6_1)
    conv_6_layer_6_2 = concatenate([batch_norm_6, layer_6_1, layer_6_2])
    layer_6_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_6_layer_6_2)
    conv_6_layer_6_3 = concatenate([batch_norm_6, layer_6_1, layer_6_2, layer_6_3])
    layer_6_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_6_layer_6_3)
    dense_block_6 = concatenate([batch_norm_6, layer_6_1, layer_6_2, layer_6_3, layer_6_4])
    dense_block_6 = Dropout(dropout_rate)(dense_block_6)
    
    deconv_1 = Conv2DTranspose(int(perc_deconv_filter * block_6_size), (2, 2), strides = (2, 2), 
                               padding = 'same', activation = 'relu')(dense_block_6)
    deconv_pool_1 = concatenate([skip_connection_5, deconv_1])
    
    batch_norm_7 = BatchNormalization()(deconv_pool_1)
    layer_7_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_7)
    conv_7_layer_7_1 = concatenate([batch_norm_7, layer_7_1])
    layer_7_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_7_layer_7_1)
    conv_7_layer_7_2 = concatenate([batch_norm_7, layer_7_1, layer_7_2])
    layer_7_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_7_layer_7_2)
    conv_7_layer_7_3 = concatenate([batch_norm_7, layer_7_1, layer_7_2, layer_7_3])
    layer_7_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_7_layer_7_3)
    dense_block_7 = concatenate([batch_norm_7, layer_7_1, layer_7_2, layer_7_3, layer_7_4])
    dense_block_7 = Dropout(dropout_rate)(dense_block_7)
    
    deconv_2 = Conv2DTranspose(int(perc_deconv_filter * block_7_size), (2, 2), strides = (2, 2), 
                               padding = 'same', activation = 'relu')(dense_block_7)
    deconv_pool_2 = concatenate([skip_connection_4, deconv_2])
    
    batch_norm_8 = BatchNormalization()(deconv_pool_2)
    layer_8_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_8)
    conv_8_layer_8_1 = concatenate([batch_norm_8, layer_8_1])
    layer_8_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_8_layer_8_1)
    conv_8_layer_8_2 = concatenate([batch_norm_8, layer_8_1, layer_8_2])
    layer_8_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_8_layer_8_2)
    conv_8_layer_8_3 = concatenate([batch_norm_8, layer_8_1, layer_8_2, layer_8_3])
    layer_8_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_8_layer_8_3)
    dense_block_8 = concatenate([batch_norm_8, layer_8_1, layer_8_2, layer_8_3, layer_8_4])
    dense_block_8 = Dropout(dropout_rate)(dense_block_8)
    
    deconv_3 = Conv2DTranspose(int(perc_deconv_filter * block_8_size), (2, 2), strides = (2, 2), 
                               padding = 'same', activation = 'relu')(dense_block_8)
    deconv_pool_3 = concatenate([skip_connection_3, deconv_3])
    
    batch_norm_9 = BatchNormalization()(deconv_pool_3)
    layer_9_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_9)
    conv_9_layer_9_1 = concatenate([batch_norm_9, layer_9_1])
    layer_9_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_9_layer_9_1)
    conv_9_layer_9_2 = concatenate([batch_norm_9, layer_9_1, layer_9_2])
    layer_9_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_9_layer_9_2)
    conv_9_layer_9_3 = concatenate([batch_norm_9, layer_9_1, layer_9_2, layer_9_3])
    layer_9_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_9_layer_9_3)
    dense_block_9 = concatenate([batch_norm_9, layer_9_1, layer_9_2, layer_9_3, layer_9_4])
    dense_block_9 = Dropout(dropout_rate)(dense_block_9)
    
    deconv_4 = Conv2DTranspose(int(perc_deconv_filter * block_9_size), (2, 2), strides = (2, 2), 
                               padding = 'same', activation = 'relu')(dense_block_9)
    deconv_pool_4 = concatenate([skip_connection_2, deconv_4])
    
    batch_norm_10 = BatchNormalization()(deconv_pool_4)
    layer_10_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_10)
    conv_10_layer_10_1 = concatenate([batch_norm_10, layer_10_1])
    layer_10_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_10_layer_10_1)
    conv_10_layer_10_2 = concatenate([batch_norm_10, layer_10_1, layer_10_2])
    layer_10_3 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_10_layer_10_2)
    conv_10_layer_10_3 = concatenate([batch_norm_10, layer_10_1, layer_10_2, layer_10_3])
    layer_10_4 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_10_layer_10_3)
    dense_block_10 = concatenate([batch_norm_10, layer_10_1, layer_10_2, layer_10_3, layer_10_4])
    dense_block_10 = Dropout(dropout_rate)(dense_block_10)
    
    deconv_5 = Conv2DTranspose(int(perc_deconv_filter * block_10_size), (2, 2), strides = (2, 2), 
                               padding = 'same', activation = 'relu')(dense_block_10)
    deconv_pool_5 = concatenate([skip_connection_1, deconv_5])
    
    batch_norm_11 = BatchNormalization()(deconv_pool_5)
    layer_11_1 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(batch_norm_11)
    conv_11_layer_11_1 = concatenate([batch_norm_11, layer_11_1])
    layer_11_2 = Conv2D(num_block_layer_filters, (3, 3), padding = 'same', activation = 'relu')(conv_11_layer_11_1)
    dense_block_11 = concatenate([batch_norm_11, layer_11_1, layer_11_2])
    dense_block_11 = Dropout(dropout_rate)(dense_block_11)
    
    pred_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(dense_block_11)
    
    mfrn_model = Model(inputs = img_input, outputs = pred_layer)
    mfrn_model.compile(loss = dice_coef_loss, optimizer = Adam(lr = l_r), metrics = [dice_coef])
    
    return mfrn_model