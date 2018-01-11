# **Behavioral Cloning**

## Writeup 

------

**Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior

- Build, a convolution neural network in Keras that predicts steering angles from images

- Train and validate the model with a training and validation set

- Test that the model successfully drives around track one without leaving the road

- Summarize the results with a written report

  [//]: # "Image References"
  [image1]: ./res/msel_vs_epoch_lenet_model_30_percent.png "MSE Loss for LeNet Model with 30% Chance to Add Augmented Data Set"
  [image2]: ./res/lenet_model_bridge_stock_30_1.png "First Try Running the Simulator"
  [image3]: ./res/lenet_model_bridge_stock_30_2.png "Second Try Running the Simulator"
  [image4]: ./res/center_lane_driving.jpg "Center Lane Driving"
  [image5]: ./res/right_to_center_driving.jpg "Back to Center from the Right Side"
  [image6]: ./res/left_to_center_driving.jpg "Back to Center from the Left Side"
  [image7]: ./res/center_lane_driving_flipped.jpg "Flipped Image"
  [image8]: ./res/input_image.jpg "Input Image - Shape = (160, 320, 3)"
  [image9]: ./res/cropped_image.jpg "Cropped Image - Shape = (90, 320, 3)"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

------

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- `model.py` containing the script to create and train the model
- `helper.py` containing the functions for loading data, training, and saving the model
- `drive.py` for driving the car in autonomous mode
- `lenet_model.h5` containing a trained convolution neural network based on LeNet architecture
- `nvidia_model.h5` containing a trained convolution neural network based on NVidia architecture
- `README.md` summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py lenet_model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for loading data set, training, and saving the convolution neural network. Also it produces and saves the _Mean Squared Error Loss (MSEL)_ for the model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I modularized different parts of the code inside `helper.py` and I used its functions inside the `model.py`.

### Model Architecture and Training Strategy

#### 1. Model architecture

I used both LeNet and NVIDIA architecture. In both models I used `ReLU` as activation function. The architecture of each model is as following:

##### LeNet Architecture

| Layer (type)                   | Output Shape        | Parameter |
| :----------------------------- | ------------------- | --------- |
| lambda_1 (Lambda)              | (None, 160, 320, 3) | 0         |
| cropping2d_1 (Cropping2D)      | (None, 90, 320, 3)  | 0         |
| conv2d_1 (Conv2D)              | (None, 86, 316, 6)  | 456       |
| max_pooling2d_1 (MaxPooling2D) | (None, 43, 158, 6)  | 0         |
| conv2d_2 (Conv2D)              | (None, 39, 154, 6)  | 906       |
| max_pooling2d_2 (MaxPooling2D) | (None, 19, 77, 6)   | 0         |
| flatten_1 (Flatten)            | (None, 8778)        | 0         |
| dropout_1 (Dropout)            | (None, 8778)        | 0         |
| dense_1 (Dense)                | (None, 120)         | 1053480   |
| dropout_2 (Dropout)            | (None, 120)         | 0         |
| dense_2 (Dense)                | (None, 84)          | 10164     |
| dropout_3 (Dropout)            | (None, 84)          | 0         |
| dense_3 (Dense)                | (None, 1)           | 85        |

> Total number of parameters for this model is 1,065,091 and all of them need to be trained. `lenet_model()` in `helper.py` builds and returns this model.

##### NVIDIA Architecture

| Layer (type)              | Output Shape        | Parameter |
| :------------------------ | ------------------- | --------- |
| lambda_1 (Lambda)         | (None, 160, 320, 3) | 0         |
| cropping2d_1 (Cropping2D) | (None, 90, 320, 3)  | 0         |
| conv2d_1 (Conv2D)         | (None, 43, 158, 24) | 1824      |
| conv2d_2 (Conv2D)         | (None, 20, 77, 36)  | 21636     |
| conv2d_3 (Conv2D)         | (None, 8, 37, 48)   | 43248     |
| conv2d_4 (Conv2D)         | (None, 6, 35, 64)   | 27712     |
| conv2d_5 (Conv2D)         | (None, 4, 33, 64)   | 36928     |
| flatten_1 (Flatten)       | (None, 8448)        | 0         |
| dropout_1 (Dropout)       | (None, 8448)        | 0         |
| dense_1 (Dense)           | (None, 100)         | 844900    |
| dropout_2 (Dropout)       | (None, 100)         | 0         |
| dense_2 (Dense)           | (None, 50)          | 5050      |
| dropout_3 (Dropout)       | (None, 50)          | 0         |
| dense_3 (Dense)           | (None, 10)          | 510       |
| dropout_4 (Dropout)       | (None, 10)          | 0         |
| dense_4 (Dense)           | (None, 1)           | 11        |

> Total number of parameters for this model is 981,819 and all of them need to be be trained. `nvidia_model()` in `helper.py` builds and returns this model.

#### 2. Reduce overfitting in the model

Both models contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. Moreover, the models were tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an `adam` optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and randomly generated augmented data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because I used it before and I got good results. Also I wanted to use NVIDIA architecture as well to see how it works compare to LeNet architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). 

First I started with **LeNet model**. Training went well and model had a low MSE with a good convergence: `loss: 0.0147 - val_loss: 0.0135` (figure below).

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. But when I ran the simulator with `lenet_model.h5`, the car stuck on the bridge (below images).

| First Try Running the Simulator | Second Try Running the Simulator |
| ------------------------------- | -------------------------------- |
| ![alt text][image2]             | ![alt text][image3]              |

It meant the model was good but I needed more training data. First thing that I did to improve the driving behavior in these cases was increasing the probability of generating the augmented data from **0.3** to **0.4**. 

> After loading each image and adding it to the data set, the code generated an augmented data with probability less than 0.5 and added it to the data set. Look at `load_data_set_one_camera()` in `helper.py`.

You might say that why you don't increase it to **1.0**. The answer is if I increase it even to **0.5** I will get the **Out of Memory Error**. However, it doesn't solve the problem because car stick somewhere else in the road. 

For gathering data I tried to keep car in middle of the rood. So, model only learns to drive car **if car remains in middle of the road** and if car gets close to the sides the chance of sticking is large. To rescue car when deflects from the center, is driving the car in the same situation. I did this for both sides (left and right) and both directions (heading forward and backward).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architectures is what I described earlier (above tables). I used `Dropout` layers to avoid overfitting. Also, I added one `Lambda` layer at the beginning for normalizing the input data and one `Cropping` Layer afterward to crop region of interest of each data for training. ***Region of interest*** *is the same image without trees at the top and car hood at the bottom.* 

| Input Image - Shape=(160, 320, 3) | Cropped Image - Shape=(90, 320, 3) |
| :-------------------------------: | :--------------------------------: |
|        ![alt text][image8]        |        ![alt text][image9]         |

You can find more info for training each model as follows:

------

##### LeNet Model

```python
loss='mse', optimizer='adam', valid_split=0.20, epochs=7
```

Train on 26980 samples, validate on 6746 samples. The speed of training was 16 (ms/step) and 400 (sec) for each epoch in average.

![alt text][image1]

##### NVIDIA Model

```python
loss='mse', optimizer='adam', valid_split=0.20, epochs=7
```

Train on 23062 samples, validate on 5766 samples. The speed of training was 16 (ms/step) and 380 (sec) for each epoch in average.

------



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center lane driving (below images).

| Back to Center from the Right Side | Back to Center from the Left Side |
| :--------------------------------: | :-------------------------------: |
|        ![alt text][image5]         |        ![alt text][image6]        |

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images thinking that this would be good if the car is driven in opposite direction. For example, here is an image that has then been flipped:).

|      Original       |       Flipped       |
| :-----------------: | :-----------------: |
| ![alt text][image4] | ![alt text][image7] |

I Also drove the car in both direction on the rood (heading backward and forward).

After the collection process, I had 20577 number of data points. I then preprocessed this data by  `cv2.flip()`  from `cv2` package randomly (40 percent chance). It added 3000~4000 more samples to the data set as augmented data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by final MSE Loss diagrams (above). I used an `adam` optimizer so that manually training the learning rate wasn't necessary.

## License

[MIT License](LICENSE).