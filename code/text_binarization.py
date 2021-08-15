#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:17:45 2020

@author: vonikakv
"""

import imageio
import matplotlib.pyplot as plt
from skimage.filters import (gaussian, threshold_otsu, threshold_niblack, 
                             threshold_sauvola)
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.morphology import remove_small_objects



plt.close('all')



def binarize_text(
        image_array, 
        sigma_surround=10, 
        sigma_center=0.5,
        remove_elements_smaller_than=10,  # pixels
        verbose=False
        ):
    '''
        ----------------------------------------------------------------------
          Adaptive document binarization based on OFF center-surround cells
        ----------------------------------------------------------------------
        Takes a text image and returns a binary version of it, based on
        activations from modeled OFF center-surround cells of the retina, 
        which detect light decrements (text) on bright background (page). 
        More information can be found in the papers:
            
        Vonikakis, V., Andreadis, I., & Papamarkos, N. (2011). Robustdocument 
        binarization with OFF center-surround cells. Pattern Analysis and 
        Applications, 14(3), 219-234.
        
        Vonikakis, V., Andreadis, I., Papamarkos, N., & Gasteratos, A. (2007). 
        Adaptive Document Binarization: AHuman Vision Approach. Int. 
        Conference on Computer Vision Theory and Applications. (pp. 104-110). 
        Barcelona, Spain.
        
        INPUTS
        ------
        image_array: numpy array HxWx3 or HxW
            The numpy array of the input image. Either RGB or grayscale.
        sigma_surround: float
            The standard deviation of the gaussian filter which is used to
            model the surround of the OFF center-surround cells.
        sigma_center: float
            The standard deviation of the gaussian filter which is used to
            model the center of the OFF center-surround cells. sigma_center 
            should be smaller than sigma_surround. Larger values minimize the
            noise in the output binary image, but they may filter out small
            details of the binary text characters. sigma_center and 
            sigma_surround should be selected with the size of the output text 
            in mind. 
        remove_elements_smaller_than: int or None
            If None, no denoising is applied. If int, all elements smaller 
            than the number specified, are removed. E.g. if 
            remove_elements_smaller_than=10 all elements smaller than 10 
            pixels will be removed.
        verbose: Boolean
            Whether or not to show visualizations for intermediate stages of 
            the algorithm and comparisons with other basic binarization 
            mathods.
        
        OUTPUT
        ------
        binary_off_cs_cells: binary numpy array HxW
            Numpy array of the binary output image.
            
        '''

    
    if len(image_array.shape) == 3:
        image = rgb2gray(image_array.copy())  # rgb-2-grayscale [0,255]->[0,1]
    else:
        image = img_as_float(image_array.copy())  # [0,255]->[0,1]
     
    # modelling center surround receptive fields as gaussians
    surround = gaussian(image, sigma=sigma_surround, mode='reflect')
    center = gaussian(image, sigma=sigma_center, mode='reflect')

    # off center-surround cell activations
    off_cs_cells = surround - center
    off_cs_cells = ((1 + surround) * off_cs_cells) / (surround + off_cs_cells)
    off_cs_cells[off_cs_cells<0] = 0  # truncate within limits
    off_cs_cells[off_cs_cells>1] = 1
    off_cs_cells = 1 - off_cs_cells  # invert for dark letters
    
    # global threshold on the off center surround cell activations
    binary_off_cs_cells = off_cs_cells > threshold_otsu(off_cs_cells)
    
    # morphological filtering: removing elements with small number of pixels
    if remove_elements_smaller_than is not None:
        binary_off_cs_cells_denoise = ~remove_small_objects(
            ar=~binary_off_cs_cells,  # black<->white
            min_size=remove_elements_smaller_than, 
            connectivity=1, 
            in_place=False
            )
    else:
        binary_off_cs_cells_denoise = binary_off_cs_cells
    
    
    
    if verbose is True:
        
        # visualize processing stages
        plt.figure(figsize=(12,7))
        plt.subplot(2,3,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Input image')
        
        plt.subplot(2,3,2)
        plt.imshow(surround, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Surround')
        
        plt.subplot(2,3,3)
        plt.imshow(center, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Center')
        
        plt.subplot(2,3,4)
        plt.imshow(off_cs_cells, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF center-surround cells')
        
        plt.subplot(2,3,5)
        plt.imshow(binary_off_cs_cells, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF center-surround binary')
        
        plt.subplot(2,3,6)
        plt.imshow(binary_off_cs_cells_denoise, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF center-surround denoised')
        
        plt.tight_layout()
        plt.show()
        
        # comparison with other basic methods (Otsu, Niblack, Sauvola)
        binary_otsu = image > threshold_otsu(image)
        window_size = 25
        thresh_niblack = threshold_niblack(
            image, 
            window_size=window_size, 
            k=0.8
            )
        binary_niblack = image > thresh_niblack
        thresh_sauvola = threshold_sauvola(
            image, 
            window_size=window_size
            )
        binary_sauvola = image > thresh_sauvola
    
        plt.figure(figsize=(12,7))
        plt.subplot(2,3,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Original image')
        
        plt.subplot(2,3,2)
        plt.imshow(binary_otsu, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Otsu global thresholding')
          
        plt.subplot(2,3,3)
        plt.imshow(binary_niblack, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Niblack local thresholding')
        
        plt.subplot(2,3,5)
        plt.imshow(binary_sauvola, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Sauvola local thresholding')
        
        plt.subplot(2,3,6)
        plt.imshow(binary_off_cs_cells_denoise, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF center-surround local thresholding')
        
        # plt.suptitle('Method comparison')
        plt.tight_layout()
        plt.show()
        

    return binary_off_cs_cells_denoise




if __name__=="__main__":
    
    # uncomment to select different test images
    filename = "../data/stain.jpg"
    # filename = "../data/shadow1.jpg"
    # filename = "../data/shadow2.jpg"
    # filename = "../data/historical.jpg"
    
    image = imageio.imread(filename)
    image_binary = binarize_text(
        image_array=image, 
        remove_elements_smaller_than=10,
        verbose=True
        )
    

    
    
    
    

