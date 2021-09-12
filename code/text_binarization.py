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



def get_off_center_surround(
    # Generates the response of OFF center-surround cells, given the receptive
    # field images of center and surround (modeled as gaussians).
        center,
        surround, 
        invert=True,
        min_max_norm = True
        ):
    
    RANGE_MAX = 1
    RANGE_MIN = 0

    # off center-surround cell activations
    off_cs_cells = surround - center
    off_cs_cells[off_cs_cells<RANGE_MIN] = RANGE_MIN
    off_cs_cells = (((RANGE_MAX + surround) * off_cs_cells) / 
                    (surround + off_cs_cells))
    
    # truncate within limits
    off_cs_cells[off_cs_cells<RANGE_MIN] = RANGE_MIN  
    off_cs_cells[off_cs_cells>RANGE_MAX] = RANGE_MAX
    
    if invert is True:
        off_cs_cells = RANGE_MAX - off_cs_cells  # invert for dark letters
    
    
    if min_max_norm is True:
        off_cs_cells = ((off_cs_cells - off_cs_cells.min()) / 
                        (off_cs_cells.max() - off_cs_cells.min()))
        
    
    return off_cs_cells
    
    
    


def binarize_text(
        image_array, 
        center_surround_sigma,
        boldness = 1.0,
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
        center_surround_sigma: list of dictionaries
            Each dictionary represents a spatial scale and needs to have 2
            specific fields:
                1. 'sigma_surround': the standard deviation of the gaussian 
                    representing the surround of the OFF receptive field.
                2. 'sigma_center' the standard deviation of the gaussian 
                    representing the center of the OFF receptive field.
            Both spatial scales depend on the resolution of text characters 
            that need to be binarized. For higher resolution text, larger
            sigmas will be needed. Center sigma usually needs to be small 
            (e.g. 0.5 or 1). Larger center sigmas will not be affected by 
            noise, but they will filter out fine details from the text. Smaller
            center sigmas, will detect finer text details, but will also 
            extract noise. The surround sigma needs to be larger than the
            center, and helps to discount illumination or stain variations. 
            You can include as many scales as you want, but usually 2 are
            enough. 
        boldness: float
            Binarization parameter that controls how bold or thin the text 
            characters will look. boldness=1.0 is the default. Values below
            1.0 (say 0.95) will make the characters thinner, whereas values
            above 1.0 (say 1.1) will make the text characters thicker. The
            upper and lower values of this parameter depend on the image. If
            the values are too large (e.g. 1.8) or too low (e.g. 0.3) then
            the image will be either all black or all white.
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
        
        

    
    # generate lists of images of the centers, the surrounds and the 
    # off-center-surround responses, the surrounds, based on the input scales
    ls_off_cs_cells = []
    ls_surrounds = []
    ls_centers = []
    for scale in center_surround_sigma:
     
        # modelling center surround receptive fields as gaussians
        surround = gaussian(
            image, 
            sigma=scale['sigma_surround'], 
            mode='reflect'
            )
        center = gaussian(
            image, 
            sigma=scale['sigma_center'], 
            mode='reflect'
            )
        
        # keep the images for later visualizations
        ls_surrounds.append(surround)
        ls_centers.append(center)

        # off center-surround cell activations
        ls_off_cs_cells.append(
            get_off_center_surround(
            center=center, 
            surround=surround, 
            invert=True,
            min_max_norm=False
            )
        )

    # combine all the off-center-surround response images
    off_cs_cells = ls_off_cs_cells[0].copy()
    for i in range(1, len(ls_off_cs_cells)):
        off_cs_cells += ls_off_cs_cells[i]
    
    # min-max normalization to bring back values to [0,1] and supress noise
    off_cs_cells = ((off_cs_cells - off_cs_cells.min()) / 
                    (off_cs_cells.max() - off_cs_cells.min()))
    
    # global threshold on the off center surround cell activations
    binary_off_cs_cells = off_cs_cells > threshold_otsu(off_cs_cells) * boldness
    
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

        total_scales = len(ls_off_cs_cells)
        
        # visualize receptive fields and off responses
        plt.figure(figsize=(12,7))
        for i in range(total_scales):
            plt.subplot(3,total_scales,i+1)
            plt.imshow(ls_surrounds[i], cmap='gray', vmin=0, vmax=1)
            plt.axis(False)
            plt.grid(False)
            plt.title('Surround ' + str(i+1))
            
            plt.subplot(3,total_scales,i+total_scales+1)
            plt.imshow(ls_centers[i], cmap='gray', vmin=0, vmax=1)
            plt.axis(False)
            plt.grid(False)
            plt.title('Center ' + str(i+1))
            
            plt.subplot(3,total_scales,i+2*total_scales+1)
            plt.imshow(ls_off_cs_cells[i], cmap='gray', vmin=0, vmax=1)
            plt.axis(False)
            plt.grid(False)
            plt.title('OFF center-surround ' + str(i+1))
        plt.tight_layout()
        plt.show()
        
        # visualize off responses and their combination
        plt.figure(figsize=(12,7))
        for i in range(total_scales):            
            plt.subplot(1,total_scales+1,i+1)
            plt.imshow(ls_off_cs_cells[i], cmap='gray', vmin=0, vmax=1)
            plt.axis(False)
            plt.grid(False)
            plt.title('OFF center-surround ' + str(i+1))
        plt.tight_layout()
        plt.show()
        
        plt.subplot(1,total_scales+1,total_scales+1)
        plt.imshow(off_cs_cells, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF combined')
        
        plt.suptitle('Comparison of OFF c-s responses')
        plt.tight_layout()
        plt.show()
        
        # visualize processing stages
        plt.figure(figsize=(12,7))
        plt.subplot(2,2,1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('Input image')
        
        plt.subplot(2,2,2)
        plt.imshow(off_cs_cells, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF center-surround cells')
        
        plt.subplot(2,2,3)
        plt.imshow(binary_off_cs_cells, cmap='gray', vmin=0, vmax=1)
        plt.axis(False)
        plt.grid(False)
        plt.title('OFF center-surround binary')
        
        plt.subplot(2,2,4)
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
    
    
    # setting spatial scales. You can add more if you need
    scale_mid = {'sigma_surround': 30, 'sigma_center': 1}
    scale_fine = {'sigma_surround': 10, 'sigma_center': 0.5}
    
    image = imageio.imread(filename)  # load image
    
    image_binary = binarize_text(
        image_array=image, 
        center_surround_sigma = [scale_fine, scale_mid], # for more scales
        # center_surround_sigma = [scale_fine],  # for single scale
        boldness = 0.9,
        remove_elements_smaller_than=10,
        verbose=True
        )
    

    
    
    
    

