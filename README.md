# Practice Problem Intel Scene Classification Challenge in Analytics Vidhya

# Problem Statement:
How do we, humans, recognize a forest as a forest or a mountain as a mountain? 
We are very good at categorizing scenes based on the semantic representation and object affinity, but we know very little about the processing and encoding of natural scene categories in the human brain. 
In this practice problem, we are provided with a dataset of ~25k images from a wide range of natural scenes from all around the world. 
My task is to identify which natural scene can the image be categorized into.

# Dataset Description:
There are 17034 images in train and 7301 images in test data.
The categories of natural scenes and their corresponding labels in the dataset are as follows -
{     'buildings' -> 0,     'forest' -> 1,     'glacier' -> 2,     'mountain' -> 3,     'sea' -> 4,     'street' -> 5 } 

 Variable 	              Definition
image_name 	     Name of the image in the dataset (ID column)
label 	          Category of natural scene (target column) 

For Further Information follow the below Link:

Link: https://datahack.analyticsvidhya.com/contest/practice-problem-intel-scene-classification-challe/

# Different Method of Image Classification:

# 1) Transfer Learning: 

   Taking the advantage of open sourced model trained on imagenet data

# 1.a) Transfer Learning Type1_ResNet50.ipynb 
       (Accuracy Level:0.932420091324201): 
ResNet50 is a 50 layer Residual Network. Keras consists of this pre-trained model. I have used this model to generate features of the images of training data set. These features have been used as a input for Multi layer Neural Netowrk model to classify the category of natural scene

# 1.b) Transfer Learning Type1_VGG16.ipynb
       (Accuracy Level:0.925570776255708):
VGG16 is 16 layers of Visual Geometry Group. I have also used this model to generate features of the images of training data set. These are used as input in neural network model. Keras has inbuilt model version.

# 2) User-Defined Model Architecture:



