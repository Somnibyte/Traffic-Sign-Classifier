# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./project_images/histogram.png "Visualization"
[image2]: ./project_images/grayscale.png "Grayscaling"
[image4]: ./newimages/new1.jpg "Traffic Sign 1"
[image5]: ./newimages/new2.jpg "Traffic Sign 2"
[image6]: ./newimages/new3.jpg "Traffic Sign 3"
[image7]: ./newimages/new4.jpg "Traffic Sign 4"
[image8]: ./newimages/new5.jpg "Traffic Sign 5"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the frequency of each class within each of our datasets (train, test, validation).

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing process starts with grascaling the input data. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image using the mini-max scaling method described in the course. I found that normalizing RGB images did not improve the accuracy of my model. The grayscale images along with the normalizing method I used greatly increased my models accuracy. 
I've also attempted to augment my data, but I did not find any significant increase in accuracy. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale Image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 1x1 stride, valid padding, outputs 5x5x32    									|
| RELU					|												|
| Fully connected		| Input = 800. Output = 120.        									|
| RELU					|												|
| Dropout					|					50% Chance			|
| Fully connected		| Input = 120. Output = 84. 									|
| RELU					|												|
| Dropout					|					50% Chance			|
| Fully connected		| Input = 84. Output = 43. 									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a learning rate of *0.001*, the Adam optimizer, a dropout rate of *0.5*, a batch size of *128* and *150* epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.95
* test set accuracy of 0.928

The approach I took to increasing the accuracy of my model consisted of both an interative approach and an architectural approach. First, I started off by utilizing the Lenet architecture as a baseline. The accuracy of my model using this architecture was between 80-85%. This accuracy was achieved by tweaking hyper parmeters such as the batch size and learning rate. I found that this initial model suffered from high variance. I identified this by comparing the accuracy of my training set with the accuracy of my validation set. I noticed that the training set error was significantly lower than the validation set error. 

A common remedy for a model that has high variance is to try to either increase your training data, apply regularization, or modify the architecture of your model. I decided to try regularizing my model by applying the dropout operation to the last two fully connected layers of my model. I experimented with the dropout rate hyper parameter and found that a dropout rate of 0.5 was sufficient. My model’s validation accuracy at this point peaked at 88%. I then continued to experiment with modifying the architecture of my model. I found that lowering the batch size, and adding a 1x1 convolution before the first fully connected layer increased my model’s validation accuracy significantly. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)      		| Speed limit (50km/h)   									| 
| Speed limit (70km/h)     			| Speed limit (70km/h) |
| Right-of-way at the next intersection					| Right-of-way at the next intersection|
| Speed limit (60km/h)	      		| Yield|
| Priority road			|Priority road	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the second to last cell of the Ipython notebook.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.90485465         			| Speed limit (50km/h)   									| 
| 1.0     				| Speed limit (70km/h) 										|
| 1.0					| Right-of-way at the next intersection											|
| 0.9993363 | Yield					 				|
| 0.6629613				    | Priority road      							|


For the first image, the model believed that the 100 km/h sign was a 50 km/h sign with a probability of 90%. The second highest probability for the 100 km/h sign was the actual 100 km/h sign with a probability of 0.09377718%. The model correctly identified the 70 km/h, and right-of-way with 100% accuracy. The model believed that the 60 km/h sign was a yield sign with a probability of 99%. Unfortunately the 60 km/h sign was not included in any of the top 5 predictions for this image. The final image, the priority road sign, was correctly identified with a probability of 66%. One hypothesis for the reason that my model failed to identify the 100km/h and 60km/h signs is that there is a smaller distribution of speed limit signs compared to other signs. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




