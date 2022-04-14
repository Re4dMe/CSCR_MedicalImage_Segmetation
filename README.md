# CSCR_MedicalImage_Segmetation" 
Central serous chorioretinopathy (CSCR) is a chorioretinal disorder of the eye characterized by serous detachment of the neurosensory retina at the posterior pole of the eye. It typically occurs in males in their 20s to 50s who exhibit acute or sub-acute central vision loss or distortion. And the diagnosis of CSCR can be made by IR or FAG imaging. In this project we apply Nested U-Net, which provides more fined-grain features than traditional U-Net by connect encoder-decoder with nested skip highpaths, to generate image segementation result automatically. 
https://www.aao.org/eye-health/diseases/what-is-central-serous-retinopathy
##Requirements
* matplotlib
* numpy
* Pillow
* torch
* torchvision
* tensorboard
* future
##Run
train.py
`python train.py`
eval.py
`python eval.py`
