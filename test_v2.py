# -*- coding: utf-8 -*-

from keras.models import load_model

from numpy import newaxis
import numpy as np
import cv2
import os
import argparse
import time
from coord import CoordinateChannel2D
from model_utils import sum_squared_error, ssim,PSNR
import tensorflow as tf
from scipy.io.matlab.mio import savemat, loadmat


if __name__ == '__main__':
         
    parser = argparse.ArgumentParser(description='lightweight-net')
    parser.add_argument("--testImagePath", type=str,dest="test_path" ,help="Path of test Images",default='./test/',action="store")
    parser.add_argument("--noisy_key",type=str,dest="noisy_key_value" ,help="noisy key",default='siddplus_valid_noisy_srgb',action="store")
    args = parser.parse_args()
    

#    model = load_model('./model/model-221.74-val_mse-0.0004--val_ssim--0.9827.hdf5',custom_objects={'sum_squared_error':sum_squared_error,'ssim':ssim,'CoordinateChannel2D':CoordinateChannel2D})
    
    model = load_model('./model/Model_v2_PSNR--37.3864.hdf5',custom_objects={'PSNR':PSNR,'sum_squared_error':sum_squared_error,'ssim':ssim,'CoordinateChannel2D':CoordinateChannel2D})
    output_path = './output_file/'
    if not os.path.exists(output_path):
       os.makedirs(output_path)
    
    testImagePath = args.test_path
    

    output_path_mat = output_path+'output_predicted.mat'
    noisy_key = args.noisy_key_value
    mat = loadmat(testImagePath)
    noisy_mat =mat[noisy_key]
    n_im, h, w, c = noisy_mat.shape
    results = noisy_mat.copy()
    for i in range(n_im):
        noisy = np.reshape(noisy_mat[i, :, :, :], (h, w, c))
    #    denoised = denoiser(noisy)
        noisy = cv2.cvtColor(noisy,cv2.COLOR_RGB2BGR)
        start_time = time.time()
    
        predictop = model.predict(np.reshape(noisy/255,(1,256,256,3)))
    #    predictop = generate_output(noisy/255,model)
        denoised = cv2.cvtColor(predictop[0]*255,cv2.COLOR_BGR2RGB)
    
        results[i, :, :, :] = denoised

        end_time = time.time()
    
        print('predicted time', end_time-start_time)

        
   
        print(i)
    res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
    savemat(output_path_mat, {res_key: results})
    print("output files saved in "+output_path_mat)
