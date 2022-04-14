import PIL.Image as Image
import png
import numpy as np

im = Image.open('/home/g08410099/Documents/MedicalImage_Project02_Segmentation/ROC/ROC_input/0132_07/gt.png')
im.convert('1').save('gt.png')

