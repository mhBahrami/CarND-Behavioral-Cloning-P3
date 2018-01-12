# **Behavioral Cloning**

### Table of Contents
[Behavioral Cloning Project](#behavioral-cloning-project)

[Files Submitted & Code Quality](#files-submitted--code-quality)

1. [Submission includes all required files and can be used to run the simulator in autonomous mode](#1-submission-includes-all-required-files-and-can-be-used-to-run-the-simulator-in-autonomous-mode)

2. [Submission includes functional code](#2-submission-includes-functional-code)

3. [Submission code is usable and readable](#3-submission-code-is-usable-and-readable)

[Model Architecture and Training Strategy](#model-architecture-and-training-strategy)

1. [Model architecture](#1-model-architecture)

2. [Reduce overfitting in the model](#2-reduce-overfitting-in-the-model)

3. [Model parameter tuning](#3-model-parameter-tuning)

4. [Appropriate training data](#4-appropriate-training-data)

[Model Architecture and Training Strategy](#model-architecture-and-training-strategy-1)

1. [Solution Design Approach](#1-solution-design-approach)

2. [Final Model Architecture](#2-final-model-architecture)

3. [Creation of the Training Set & Training Process](#3-creation-of-the-training-set--training-process)

[Recording and Building the Video](#recording-and-building-the-video)

- [Recording Video in Autonomous Mode](#recording-video-in-autonomous-mode)

- [Using video.py](#using-videopy)

- [The video](#the-video)

[License](#license)

---

### Behavioral Cloning Project

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior

- Build, a convolution neural network in Keras that predicts steering angles from images

- Train and validate the model with a training and validation set

- Test that the model successfully drives around track one without leaving the road

- Summarize the results with a written report

  [//]: # "Image References"
  [image1]: ./res/msel_vs_epoch_nvidia_model.png "MSE Loss for LeNet Model with 30% Chance to Add Augmented Data Set"
  [image2]: ./res/lenet_model_bridge_stock_30_1.png "First Try Running the Simulator"
  [image3]: ./res/lenet_model_bridge_stock_30_2.png "Second Try Running the Simulator"
  [image4]: ./res/center_lane_driving.jpg "Center Lane Driving"
  [image5]: ./res/right_to_center_driving.jpg "Back to Center from the Right Side"
  [image6]: ./res/left_to_center_driving.jpg "Back to Center from the Left Side"
  [image7]: ./res/center_lane_driving_flipped.jpg "Flipped Image"
  [image8]: ./res/input_image.jpg "Input Image - Shape = (160, 320, 3)"
  [image9]: ./res/cropped_image.jpg "Cropped Image - Shape = (90, 320, 3)"
  [image10]: ./res/steering_data_distribution.png "Steering Data Distribution"
  [image11]: ./res/processed_steering_data_distribution.png "Processed Steering Data Distribution.png"

### 

------

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- [`model.py`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
- [`helper.py`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py) containing the functions for loading data, training, and saving the model
- [`drive.py`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
- [`nvidia_model.h5`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/nvidia_model.h5) containing a trained convolution neural network based on NVidia architecture
- [`README.md`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/README.md) summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py nvidia_model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for loading data set, training, and saving the convolution neural network. Also it produces and saves the _Mean Squared Error Loss (MSEL)_ for the model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I modularized different parts of the code inside `helper.py` and I used its functions inside the `model.py`.

To run the code type `%run model.py` inside a *jupyter notebook* or `python model.py` inside the terminal.

### Model Architecture and Training Strategy

#### 1. Model architecture

I used [NVIDIA architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I used `ReLU` as activation function. The architecture of this model is as following:

##### NVIDIA Architecture

| Layer (type)                             | Output Shape        | Parameter |
| :--------------------------------------- | ------------------- | --------- |
| [Cropping2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L146) | (None, 90, 320, 3)  | 0         |
| [Lambda](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L151) | (None, 90, 320, 3)  | 0         |
| [Conv2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L197) | (None, 43, 158, 24) | 1824      |
| [SpatialDropout2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L198) | (None, 43, 158, 24) | 0         |
| [Conv2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L199) | (None, 20, 77, 36)  | 21636     |
| [SpatialDropout2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L200) | (None, 20, 77, 36)  | 0         |
| [Conv2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L201) | (None, 8, 37, 48)   | 43248     |
| [SpatialDropout2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L202) | (None, 8, 37, 48)   | 0         |
| [Conv2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L203) | (None, 6, 35, 64)   | 27712     |
| [SpatialDropout2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L204) | (None, 6, 35, 64)   | 0         |
| [Conv2D](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L205) | (None, 4, 33, 64)   | 36928     |
| [Flatten](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L206) | (None, 8448)        | 0         |
| [Dropout](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L207) | (None, 8448)        | 0         |
| [Dense](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L208) | (None, 100)         | 844900    |
| [Dropout](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L209) | (None, 100)         | 0         |
| [Dense](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L210) | (None, 50)          | 5050      |
| [Dropout](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L211) | (None, 50)          | 0         |
| [Dense](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L212) | (None, 10)          | 510       |
| [Dropout](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L213) | (None, 10)          | 0         |
| [Dense](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L214) | (None, 1)           | 11        |

> Total number of parameters for this model is 981,819 and all of them need to be be trained. [`nvidia_model()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L185) in `helper.py` builds and returns this model.

#### 2. Reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. `dropout_rate` for `SpatialDropout2D` and `Dropout` layers has been set to **0.2** and **0.5** respectively.

The model was trained and validated on different data sets to ensure that the model was not overfitting. Moreover, the models were tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an [`Adam()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L220) optimizer with a learning rate equal to **0.001**. The number of [`epochs`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/model.py#L34) is **10** and the [`batch_size`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L227) is **32**. I used [`mse`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/model.py#L33) as loss function and set [`valid_split`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/model.py#L34) to **0.2**.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and randomly generated augmented data by flipping the original image.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA. In order to gauge how well the model was working, I split the generated data set for images and steering angle data into a training (80%) and validation set (20%). 

I used all data including center, left,  right, and flipped version of each to train the model. Training went well and model had a low MSE with a good convergence: `loss: 0.0320 - val_loss: 0.0254`.

The final step was to run the simulator to see how well the car was driving around track one. But when I ran the simulator with `nvidia_model.h5`, the car stuck on the bridge (*below images*).

| First Try Running the Simulator | Second Try Running the Simulator |
| ------------------------------- | -------------------------------- |
| ![alt text][image2]             | ![alt text][image3]              |

It meant the model was good (because of convergence diagram for training and validation datasets) but the model doesn't really learned everything. **My first guess was for training data. ** So, lets take a look at them. At the time of reading `driving_log.csv` file I looked at the distribution of `steering` values (*below image - left*). The number of images with `steering=0.0` is ~12,000. It means that the model mostly learns to drive the car in situation with zero steering. And if stuck somewhere on the sides probably model doesn't know what to do! 

So, I dropped around 10,000 of them randomly (*below image - right*).  In this case the distribution of steering data set looks like a normal distribution around the 0 with a larger variance of steering lane angle. This helps to avoid over fitting the model around data set samples with a zero steering value.

> Look at the [`load_data_set_log()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L60) and [`drop()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L51) functions in `helper.py`.

| Distribution of Steering for First Data Set | Distribution of Modified Data Set |
| :--------------------------------------: | :-------------------------------: |
|           ![alt text][image10]           |       ![alt text][image11]        |

Moreover, I didn't used every image even after dropping. For each frame there is **3 different version** for center, left, and right camera. I only used one out of three for each frame randomly. The chance of picking center-camera frame was [**50 percent**](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L105) and the other 2 frames **25** percent  each ([here](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L111) & [here](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L117)).

> Look at [`load_data_set()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L93) in `helper.py`.

If the left-camera frame or the right-camera frame is selected the steering lane angel must be adjusted. I randomly selected a number between [`[0.20, 0.25]`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L79) Then, I added it to and subtracted it from the steering value for the left and right frames respectively. In addition, I flipped the selected image to data set by [**50%** chance](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L86) to data set. To calculate steering angle for flipped frame I only multiplied the original frame's steering value by [`-1.0`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L88).

> Look at [`load_camera_data_set()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L69) in `helper.py`.

I also increased the *epoch* from **7** to **10** and decreased *learning rate* from **0.01** to **0.001**.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

> **Note**
>
> There were multiple tries to find the best values. you can see them in [`helper.ipynb`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.ipynb) file.

#### 2. Final Model Architecture

The final model architectures is what I described earlier (*above table*). I used `Dropout` layers to avoid overfitting. Also, I added one `Cropping` Layer at the beginning to extract region of interest of each frame and one `Lambda` layer afterward for normalizing the cropped frame. ***Region of interest*** *is the same image without trees at the top and car hood at the bottom.* 

| Input Image - Shape=(160, 320, 3) | Cropped Image - Shape=(90, 320, 3) |
| :-------------------------------: | :--------------------------------: |
|        ![alt text][image8]        |        ![alt text][image9]         |

You can find more info for training each model as follows:

------

##### NVIDIA Model

```python
>> Training the model...
>> Info: loss=mse, optimizer=<keras.optimizers.Adam object at 0x7f00c3848160>, valid_split=0.20, epochs=10
Train on 16748 samples, validate on 4188 samples
Epoch 1/10
16748/16748 [====================] - 307s 18ms/step - loss: 0.0437 - val_loss: 0.0298
Epoch 2/10
16748/16748 [====================] - 303s 18ms/step - loss: 0.0376 - val_loss: 0.0281
Epoch 3/10
16748/16748 [====================] - 308s 18ms/step - loss: 0.0352 - val_loss: 0.0261
Epoch 4/10
16748/16748 [====================] - 317s 19ms/step - loss: 0.0346 - val_loss: 0.0256
Epoch 5/10
16748/16748 [====================] - 315s 19ms/step - loss: 0.0335 - val_loss: 0.0260
Epoch 6/10
16748/16748 [====================] - 314s 19ms/step - loss: 0.0329 - val_loss: 0.0255
Epoch 7/10
16748/16748 [====================] - 316s 19ms/step - loss: 0.0322 - val_loss: 0.0285
Epoch 8/10
16748/16748 [====================] - 311s 19ms/step - loss: 0.0326 - val_loss: 0.0250
Epoch 9/10
16748/16748 [====================] - 313s 19ms/step - loss: 0.0320 - val_loss: 0.0253
Epoch 10/10
16748/16748 [====================] - 328s 20ms/step - loss: 0.0320 - val_loss: 0.0254
```

Train on 16,784 samples, validate on 4,188 samples. The speed of training was 18 (ms/step) and ~310 (sec) for each epoch in average.

![alt text][image1]

------



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center lane driving (*below images*).

| Back to Center from the Right Side | Back to Center from the Left Side |
| :--------------------------------: | :-------------------------------: |
|        ![alt text][image5]         |        ![alt text][image6]        |

To augment the data set, I also flipped images thinking that this would be good if the car is driven in opposite direction. For example, here is an image that has then been flipped.

|      Original       |       Flipped       |
| :-----------------: | :-----------------: |
| ![alt text][image4] | ![alt text][image7] |

I Also drove the car in both direction on the rood (heading backward and forward).

After the collection process, I had ~15,000 number of data points. I then preprocessed this data by  [`cv2.flip()`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/helper.py#L87)  from `cv2` package randomly (50% chance). It added ~6,000 more samples to the data set as augmented data. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by final MSE Loss diagrams (*above MSE LOSS diagram*). 

### Recording and Building the Video 

#### Recording Video in Autonomous Mode

First, I recoded frames with the following commands using [`drive.py`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/drive.py):

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists. 

#### Using video.py

Then, I generated a *60fps* video using [`video.py`](https://github.com/mhBahrami/CarND-Behavioral-Cloning-P3/blob/master/video.py):

```sh
python video.py run1
```

#### The video

You can watch the built video [here](https://youtu.be/LbQUHrNF8T4).

## License

[MIT License](LICENSE).