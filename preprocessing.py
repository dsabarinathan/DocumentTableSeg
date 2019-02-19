# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:19:43 2019

@author: sabari
"""
from skimage.filters import threshold_sauvola


def sauvola_thresholding(grayImage_,window_size=15):
    
    """"
    Sauvola thresholds are local thresholding techniques that are 
    useful for images where the background is not uniform, especially for text recognition
    
    grayImage--- Input image should be in 2-Dimension Gray Scale format
    window_size --- It represents the filter window size 
    
    """
    thresh_sauvolavalue = threshold_sauvola(grayImage_, window_size=window_size)

    thresholdImage_=(grayImage_>thresh_sauvolavalue)
    
    return thresholdImage_