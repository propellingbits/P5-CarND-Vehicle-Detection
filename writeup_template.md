##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

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
[image8]: ./examples/car-notcar.png
[image9]: ./examples/car-notcar-YCrCb.png
[image10]: ./examples/sliding-win-search-1.png
[image11]: ./examples/heat-maps.png
[image12]: ./examples/before-label.png
[image13]: ./examples/after-label.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell # 20 of my IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image8]

I then explored different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the output looks like. Parameters were altered in cell # 91.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image9]

####2. Explain how you settled on your final choice of HOG parameters.

I have not settled on params yet. It is a journey. There is still a lot pending to achieve. The current output seems to be reasonably presentable so I thought of stopping my experiments and proceed with project submission. 

RGB images achieved test accuracy of 96% while YCrCb images achieved it little over 99% so no debate about which one to choose. We choose quality of results over training time.

I have listed down params for both RGB and YcrCb images below along with my comment which explains reasoning for choosen values -
```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # as per HOG paper, it works until this limit
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL" . Works better at 'all'
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

Result-
162.09003233909607  seconds to train a classifier...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
55.07 Seconds to train SVC...
Test Accuracy of SVC =  0.9927
```

```
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

Result-
71.78193712234497  seconds to train a classifier...
Using: 6 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 1992
12.88 Seconds to train SVC...
Test Accuracy of SVC =  0.9696
```
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Please refer to code cell # 96 in my notebook for code location.

Steps for training a classifier -
1. We first convert an image to selected color space
2. Then we extract spatial and histogram features
3. At this step, we take out the HOG features
4. The above order must be maintained through out the entire pipeline otherwise
   you will see unexpected results
5. Once all extraction is done then we concatenate all of it together and return     it back to caller
6. At this point, we create vertical and horizontal arrays for features and          labels set
7. At the very end, we call svc.fit(x_train, y_train) to create a classifier
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Please refer to cell numbers - 55, 73 and 83

We tried a lot of parameteres and ultimately settled on which gave us acceptable results under faster times. The approach that we took is straight from lesson # 31. Using HOG means taking small blocks of images and searching for it across the defined area in the search window. We are also ignoring upper half of search window as cars are not yet flying :-). This also gave us speed bump. 

![alt text][image10]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Most of our approaches are straight from Ryan's youtube video.

find_cars method in lesson # 34 brings in big improvement. Instead of doing HOG transformation for each scanned part on search window, we are converting the whole target image at the beginning.

![alt text][image11]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)[](./test_video_output.mp4) [](./project_video_2 output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Please refer to following methods in cell # 84.

I recorded the positions of positive detections in each frame of the video. For more details, please refer to find_cars method at line # 87 (I just noticed a small enhancement here. There is no need to draw rectangle in this method and return an image). I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I used draw_labeled_bboxes to draw thresholded labels on the image.  

### Before and after images of appling threshold on labels:

![alt text][image12] ![alt text][image13]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

John chen's videos from both p4 and p5 are big inspirations. I will be working on to get something mind blowing like that. 

