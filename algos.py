import cv2
import numpy as np
import pandas as pd

def fourier_sharpness(img):
    fourier = np.fft.fft2(img)
    fourier_shift = np.fft.fftshift(fourier)
    magnitude_spectrum = np.abs(fourier_shift)
    max = np.amax(magnitude_spectrum)
    threshold_count = np.count_nonzero((magnitude_spectrum > max / 1000) & (magnitude_spectrum < max))    
    return threshold_count/magnitude_spectrum.size  #image quality measure and magnitude_spectrum

def canny_sharpness(img):
    edges = cv2.Canny(img, 100, 200)
    return np.count_nonzero(edges)/edges.size  #proportion of pixels that are defined as an edge

def brightness(img):
    pixelbrightness = 0
    for x in img:
        for y in x:
            pixelbrightness += (y[0]+y[1]+y[2]) / 256          
    return  pixelbrightness/(3*(len(img)*len(img[0])))

def run(img):
    d = {'fourier': fourier_sharpness(img), 'canny': canny_sharpness(img), 'brightness' : brightness(img)}
    return d

def weather(img):
    pass
