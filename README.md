## Project 5 - Vehicles Detection 
### Writeup


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
[image1]: ./test_images/car_nocar.JPG
[image2]: ./test_images/hog_extract.JPG
[image3]: ./test_images/zones.jpg
[image4]: ./test_images/scales.JPG
[image5]: ./test_images/search_grid.JPG
[image6]: ./test_images/classifier_performance.JPG
[image7]: ./test_images/heat_demo.JPG
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the notebook called 'P5-sandbox.ipynb'
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `gray` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

During training with the classifier I tried several parameters and settled for the following list:
```
color_space = 'YCrCb'   
orient = 9  # HOG orientations  
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the notebook file `P5-Classifier.ipynb` I trained a SVM classifier using a dataset adapted from the one provided by udacity. I manually selected images from the time-series data and then randomly fliped car images to make the two classes the same size, thus becoming more balanced. I used sklearn's StandardScaler to normalize the features.   
I used a randomized search CV to find a good C parameter for the LinearSVC. I tried others non-linear kernels but given the number of features I opted for a much faster LinearSVC, obtaining a final accuracy on the test set of 0.9919.   
Finally I dumped the classifier and the feature extraction parameters to pickle files for later use.

```
115.52 Seconds to find parameters...
The best parameters are {'C': 4.9770235643321135} with a score of 0.98
4.83 Seconds to train SVC...
Test Accuracy of SVC =  0.9919
My SVC predicts:  [ 1.  0.  0.  0.  0.  1.  1.  0.  1.  1.]
For these 10 labels:  [ 1.  0.  0.  0.  0.  1.  1.  0.  1.  1.]
0.004 Seconds to predict 10 labels with SVC
``` 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

All the code regarding this section can be found in the notebook called `P5-Sliding-Windows.ipynb`.   
My strategy for this was defining several zones and window scales to search for cars. Each zone had a diferent size and position, as well as different window scales.

![alt text][image3]
![alt text][image4]

Here's an image of the final search grid.

![alt text][image5]

After some experimentation I decided to use 0.7 overlap in every search zone.   
Resulting in the following window cont:

`Zone 1: 97`  
`Zone 2: 179`   
`Zone 3: 170`   
`Zone 4: 128`   
`Total windows: 574`


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I thresholded the decision function for each zone separately. This allowed for a lot of flexibity in optimizing the pipeline.
Heres' how the pipeline worked on the test images:

![alt text][image6]
![alt text][image7]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code regarding this section can be found in the file `P5-Video-Pipeline.ipynb`.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.   
I also impletemented a system to a average the bounding boxes of successive detections, you can find this inside the `draw_labeled_bboxes()` function definition.   
Due to time restriction I didn't manage to do proper testing on the pipeline and went straight to the final output.   
After several atempts, I changed a lot of my original parameters, zones and scales to get satisfactory results.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had some trouble with the hog subsampling method, because the code given in the lessons didn't quite perform as I expected. I decided to proceed without it, despite the performance cost.   
I found out that this is a very test intensive process and that it's very important to make time for creating the necessary testing pipelines. Initially I was going for a lot of window density in the search grid, but I found out later that it was overkill. In the final video pipeline I ended up eliminating one search zone and probably more than half of the windows.    
Thresholding the decision function also proved to be quite useful in eliminating false positives.   
This pipeline needs to be much more robust in order to work in a real situation. It's still very sensitive to false positives and, maybe more work in the training dataset is required (Garbage In - Garbage Out).

