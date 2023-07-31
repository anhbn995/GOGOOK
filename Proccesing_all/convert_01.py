#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:19:39 2021

@author: skm
"""


def get_min_max_image(file_path):
    ds = gdal.Open(file_path,  gdal.GA_ReadOnly)
    numband =  ds.RasterCount
    dict_band_min_max = {1:0}
    for i in range(4):
        print(dict_band_min_max)
        band = ds.GetRasterBand(i + 1)
        min_train, max_train, _, _ = band.GetStatistics(True, True)
        dict_band_min_max.update({ i+1 : {"min": min_train, "max":max_train}})
    return dict_band_min_max, numband