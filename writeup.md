## Writeup Template


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
[image1_1]: ./examples/car.png
[image1_2]: ./examples/notcar.png
[image2]: ./examples/hot_ex.png
[image3]: ./examples/slidingwindows.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/theheat.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  





### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in the code cell 2,3,4 of the `Solution.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
- ![alt text][image1_1]
- ![alt text][image1_2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using HOG parameters of `PIX_PER_CELL = 8, CELL_PER_BLOCK = 2, ORIENT = 9`:


![alt text][image2]

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters and finally chose the HOG parameter as:
- orient_=9
- pix_per_cell_=8
- cell_per_block_=2

#### 3. Train a classifier using your selected HOG features

First, I extracted car_features and notcar_features from the data set. Then processed the features with labels. After that, I split and randomized the data for training. The training function in code cell 10 is a calibrated linear Support-Vector-Classifier that takes training data and test data as input.

### Sliding Window Search

#### 1. Sliding window search.

In order to make reasonable detection, I only made sliding windows that covered the road, which are at the lower part of the picture:

![alt text][image3]

#### 2. Test Image

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

Here's a [link to my video result](./project_video_soln.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are integrated heatmaps:

![alt text][image5]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### Problems / issues.

- The pipeline still cannot 100% detect cars in the frame.
- In some situations, especially when there are shadows, non-vehicle objects will be detected as vehicles
