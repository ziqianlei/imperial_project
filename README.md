# Optical Aberration Identification and Correction #
This program was for my MSc three months project. It built a CNN model for optical aberration identification and correction.
## Introduction ##
This project built several convolutional neural networks, abCNN, for aberrations analysis of single-bead images. We demonstrated that abCNN can predict aberrations of a stack of single-bead experimental wide-field images, including defocus, astigmatism, coma and spherical aberration. Therefore, the abCNN allows futher measurements and analysis through single-bead emission patterns. With the aberration predictions from abCNN, we trained another neural network to measure z-positions of beads images with known aberrations. Aberration performances of three different microscopes have been assessed by predictions from abCNN as well.  
![overview](https://github.com/ziqianlei/imperial_project/blob/main/images/overview.png)
## abCNN architecture ##
The input example is a stack of PSF images with size⁡11×32×32. Our abCNN model consists of three individual convolution blocks, two types of residual blocks, a flatten layer, and a convergent Dense layer with 18 outputs.  
![abCNN_architecture](https://github.com/ziqianlei/imperial_project/blob/main/images/abCNN_architecture.png)
## Applications ##
### Z-position measurement ###
* Step 1: Aberration prediction by abCNN
* Step 2:PSF simulation]  
![PSF_simulation](https://github.com/ziqianlei/imperial_project/blob/main/images/PSF_simulation.png)  
* Step 3: z-position prediction 
### Objectives assessment ###
In this work, with the assistance of our trained abCNN, the aberrations of images are compared and analysed when imaging with various objectives. Three different types of objectives have been employed in this project. If you are interested in the details, please feel free to email me zx_lei@qq.com.

## Acknowledgements ##
The project modified CNN models proposed by Zhang et al.[1], and proposed two applications of this model.
[1] Zhang, P., Liu, S., Chaurasia, A. et al. Analyzing complex single-molecule emission patterns with deep learning. Nat Methods 15, 913–916 (2018). https://doi.org/10.1038/s41592-018-0153-5  
The relevant MSc dissertation is available for request.  
