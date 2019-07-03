# Resnet U-net based Scanned Document Table Segmentation

**Overview**

**Data**

ICDAR 2013 Table Competition you can download the train and test data from the below link.

https://roundtrippdf.com/en/downloads/

**Pre-processing**

The dataset contains 27 PDFs, first step is to convert the PDF into images. When we trained the model with original RGB images the segmented result is not good. Applied the distance transform on RGB images before passing into the model. It improved the overall segmentation. 



**Model**

![alt text](https://github.com/sabaridsn/DocumentTableSeg/blob/master/Presentation1.jpg)

