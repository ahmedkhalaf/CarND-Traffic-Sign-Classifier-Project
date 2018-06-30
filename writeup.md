# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./workbook_plots/training_validation_histogram.png "Class occurances in Training and Validation set"
[image2]: ./workbook_plots/training_samples.png "Samples from training set"
[image3]: ./workbook_plots/augmented_images.png "Augmented data and originals"
[image4]: ./testimgs_orig/14_Stop.jpg "14 Stop"
[image5]: ./testimgs_orig/18_General_caution.jpg "18 General caution"
[image6]: ./testimgs_orig/27_Pedestrians.JPG "27 Pedestrians"
[image7]: ./testimgs_orig/31_Wild_animals_crossing.jpg "Wild animals crossing AND 60Km/h"
[image8]: ./workbook_plots/learning_over_epochs.png "Accuracy over EPHOC Phases"
[image9]: ./workbook_plots/img0_fm0.png "First layer Feature map"
[image10]: ./workbook_plots/img1_fm0.png "First layer Feature map"
[image11]: ./workbook_plots/img2_fm0.png.png "First layer Feature map"
[image12]: ./workbook_plots/img0_fm1.png "Second layer Feature map"
[image13]: ./workbook_plots/img1_fm1.png "Second layer Feature map"
[image14]: ./workbook_plots/img2_fm1.png "Second layer Feature map"
[image15]: ./workbook_plots/img0_fm2.png "Third layer Feature map"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ahmedkhalaf/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the "len()" python function and numpy shape to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique classes/labels = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training images and validation are distributed among classes.

From the below plot, it's clear that some classes are under-represented in the training data.
Also, it's noteworthy to mention that percentage for a certain class is similar in both the training and validation data sets.
For example: "Speed limit 20Km/h" is only 1% of training images while "Speed limit 30Km/h" is about 5.7%, the precentage is similar in training and validation data sets.

![alt text][image1]

Looking into few examples from random classes, it's clear that many images are quite similar and could be seen as slight transformations to one image

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Pre-reprocessing, Balancing Population and Augmenting data

As a first step, I decided to convert the images to grayscale because this helps reduce processing and memory requirements.

I decided to generate additional data to help balance the training data set in order to prevent over-fitting.
To add more data to the the data set, I used the following [transformations mentioned in the Traffic Sign Recognition with Multi-Scale Convolutional Networks paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) :

>**Samples are randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees)**

Adding 119 - 160 images for 26 classes which sample count was less than statistical mean.
Augmented training examples 4078

Here some examples of an augmented image and the corresponding original image:

![alt text][image3]

In addition, a balanced sub-set of training data was generated as show in iPython cell #4

#### 2. Model architecture overview

My final model consisted of the following layers:
Please refer to the notebook cell #12 for exact dimensions

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  			| 
| Max pooling	   	| 1x1 stride, VALID padding		|
| RELU					|												|
| Max pooling	   	| 2x2 stride, VALID padding		|
| Convolution 3x3	| 1x1 stride, VALID padding 	|
| RELU					|												|
| dropout		|		drop 20%							|
| Max pooling	   	| 2x2 stride, VALID padding		|
| Convolution 3x3	| 1x1 stride, VALID padding 	|
| RELU					|												|
| dropout		|		drop 50%							|
| Fully connected		|        									|
| RELU					|												|
| dropout		|		drop 20%							|
| Fully connected		|     classification layer   									|
| Softmax				|        									|
|						|												|
 


#### 3. Training and Hyper Parameters

To train the model, I used a relatively small learning rate of 0.0005 suitable for training over a longer number of EPOCHs (typically 1000).

BATCH_SIZE was set to 500, also TensorFlow built-in Adam optimizer was used instead of gradient decsent.

Hyper parameters were set to
* mu = 0
* sigma = 0.1

#### 4. Model architecture development, training and Accuracy improvement

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.1%
* test set accuracy of 93.9%

An iterative approach was chosen as follows:
* LeNet architeture was used as an initial step due to its string capability to classify "glyph"s with high accuracy which is quite similar concept to symbols displayed on traffic signs, only number of classes was adjusted to 43 instead of 10, the network couldn't achieve more than 90% validation accuracy and much less in test set accuracy.
* It seems LeNet architeture couldn't capture details sufficient to classify 43 classes and at the same time overfitted to training set quickly.
* Also, the model was underfitting when normalized images were used accuracy couldn't exceed 60% in neither in training nor valdiation sets.
* In order to capture more details, more convolutional layers were added with more max pooling layers in between because maxpool is operation is less agressive. while convolutional layers help the network tolerate different insegnificant aspects including distortion, translation and rotation ..etc  With this, the model achieved 92% validation accuracy.
* An important aspect was using several dropout layers in order to prevent overfitting which was key to exceed 92% validation accuracy.
* Due to memory and available data limitations, a training epoch was broken into three phases to help counter under-represented classes:
** Augmented data
** Balanced data
** Training data
* Training using Augmented data and training sub-set with balanced classes helped the model generalize well and eventually achieve 93.9% over long number of EPOCH phases as shown in the plot below:

![alt text][image8]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I looked for traffic signs photos from [Wuppertal, Germany](https://wuppertal.de/) where I'm located now.
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7]


Pedestrians sign image might be really difficult to classify because it's circular, normally it's triangular. (external sign shape is show in visualizations for the feature maps).


Also, stop sign image shows some distortion as someone wrote on the sign using black marker or a similar thing.


Finally, one image has two signs "Speed Limit 60" and "Wild Animal Crossing" very close to each other, therefore the speed limit sign is partially included in the 32x32 image for "Wild Animal Crossing" sign.

Extracted [32x32 images can be found here](https://github.com/ahmedkhalaf/CarND-Traffic-Sign-Classifier-Project/tree/master/testimgs32x32)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing      		| Wild animals crossing   									| 
| Pedestrians     			| Keep right 										|
| Speed limit (60km/h)					| Speed limit (60km/h)											|
| Stop	      		| Stop					 				|
| General Caution			| General Caution      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

This compares favorably to the accuracy on the test set of 93.9% (and 96.1% on validation set).

It's noteworthy to mention that 32x32 image preparation can affect the network's capability to classify subject signs correctly, scaling down an image from higher resolution affects accuracy negatively.

Therefore, it might be better to use high resolution images taken from far distance than from closer distance whith the sign covering huge area when the network input size (or increase network input size if possible).

This note could be important in a future phase related to car camera setup and integration with the network.

#### 3. Softmax probabilities for Predicting on new images for each prediction.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a Wild animals crossing sign (probability of 97.2%) which is coming first in the top five soft max probabilities as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.972 | Wild animals crossing|
| 0.028 | Double curve|
| 0.000 | Dangerous curve to the left|
| 0.000 | Road narrows on the right|
| 0.000 | Right-of-way at the next intersection|


For the second image, the model is not quite sure about the sign which is classified as Keep right sign (with only 43.4%). This is probably due to the unexpected circular shape.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.434 | Keep right|
| 0.236 | Speed limit (30km/h)|
| 0.094 | End of all speed and passing limits|
| 0.052 | No vehicles|
| 0.045 | Go straight or right|

The model was relatively sure about the third image classifying it as Speed limit (60km/h) with 65.9% probability, the image actually is a Speed limit (60km/h) sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.659 | Speed limit (60km/h)|
|0.152 | Speed limit (80km/h)|
|0.148 | No passing for vehicles over 3.5 metric tons|
|0.018 | Speed limit (30km/h)|
|0.017 | Go straight or left|

Despite the distortion of stop sign in the fourth image, The model classified it successfully as Stop sign with very high certainity.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 | Stop|
| 0.000 | Speed limit (30km/h)|
| 0.000 | Traffic signals|
| 0.000 | Right-of-way at the next intersection|
| 0.000 | General caution|

Fith image was for a General caution sign, it was also classified correctly with high certainity.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000 | General caution|
| 0.000 | Traffic signals|
| 0.000 | Bumpy road|
| 0.000 | Pedestrians|
| 0.000 | Keep right|



### Visualizing the Neural Network (See Step 4 of the Ipython notebook output for more details)

The network seems to extract features related to shape, edges and internal content of the sign.
Below, the network different "filters" or feature maps for three signs different in shape and symbol/content of the sign

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
