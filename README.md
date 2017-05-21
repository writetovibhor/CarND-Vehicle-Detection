# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in line 26 through 43 of `process.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and color spaces and settled on YCrCb and performed HOG extraction on Y channel.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used `GridSearchCV()` to select the best combination of `C` and `kernel` for SVC classifier, code for grid serach and training is contained in line 166 through 182 of `process.py`. I used `C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]` and `kernel = ['linear', 'rbf']` for grid search. The algorithm selected `C=1.0` and `kernel='rbf'` as the best parameters.

Classifier training code is contained in line 180 of `pipeline.py`. I used YCbCr color space and created features using spatial binning of `(64, 64)` plus color histogram of `64` bins plus HOG extraction from Y channel, code for feature extraction is contained in line 116 through 164 of `pipeline.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Code for sliding window search at various scales is contained in lines 351 through 382 of `pipeline.py`. I used scale sizes of `1, 1.5625, 2.25, 4` and came up with following regions for sliding window search. I used `cells_per_step = 2` and window size of `64` so that there an overlap of 75% between the windows.

**`Scale = 1.0`**

![example image](./examples/scale-1.jpeg)

**`Scale = 1.5625`**

![example image](./examples/scale-1.5625.jpeg)

**`Scale = 2.25`**

![example image](./examples/scale-2.25.jpeg)

**`Scale = 4`**

![example image](./examples/scale-4.jpeg)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![example image](./examples/test-images.jpeg)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_vides/project_video.mp4).

[![video on youtube](https://img.youtube.com/vi/7RrqpCHkOd0/0.jpg)](https://www.youtube.com/watch?v=7RrqpCHkOd0)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected, code for this step is contained in lines 309 through 394 of `pipeline.py`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are ten frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First issue with this approach is performance, real-time solution will require use of GPU for doing most of the computation and scikit-learn will need to be replaced with some other library that supports GPU.

Another problem I found is that if two cars are very close to each other or behind one-another then those are detected as single object due to overlapping bounding boxes. We can use perspective transform to identify cars in different lanes and sperate out the bounding boxes.

One more issue is jumping of bounding box size, this can be reduced by averaging box size over few frames.
