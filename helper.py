import pandas as pd
import cv2
import zipfile
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


def unzip(path_to_zip_file, directory_to_extract_to='.'):
    """
    Helper function to unzinp a zip file
    """
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()
    
    
def read_csv_df(path='./data/driving_log.csv'):
    """
    Helper function to read a csv file
    """
    df = pd.read_csv(path)
    return df


def steering_drop_condition(steering):
    """
    Defines the condition for dropping a frame
    """
    return steering.values == 0.0


def get_drop_indices(data, drop_condition, keep_count):
    """
    Returns indices that should be dropped
    """
    all_indices = np.arange(len(data.values))
    indices = all_indices[drop_condition(data)]
    keep_count = len(indices) if keep_count >= len(indices) else keep_count
    
    ind_drop = shuffle(indices)[:-keep_count]
    return ind_drop


def drop(df):
    """
    Returns new data set info after dropping
    """
    indices = get_drop_indices(df.steering, drop_condition=steering_drop_condition, keep_count=1000)
    df_dropped = df.drop(indices).reset_index()
    return df_dropped


def load_data_set_log(path, dropping = True):
    """
    Loads the final data set log info
    """
    data_set_log = read_csv_df(path=path)
    if(dropping): data_set_log = drop(data_set_log)
    return data_set_log


def load_camera_data_set(X_train, y_train, path, steering, camera='c'):
    """
    Loads and adds a RGB image and its flipped to data set as FEATURES
    Loads and adds a steering value and its additive inverse to data set as MEASUREMENTS
    """
    correction = {'c':0.0, 'l':1.0, 'r':-1.0}
    
    #print(path)
    image_bgr = cv2.imread(path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _steering = float(steering) + correction[camera] * random.uniform(0.20, 0.25)
    X_train.append(image)
    y_train.append(_steering)
    
    _probability = random.uniform(0.0, 1.0)

    # Randomely add some augmented data with a 50% chance
    if(random.uniform(0.0, 1.0) <= 0.50):
        augmented_image = cv2.flip(image, 1)
        augmented_steering = _steering * (-1.0)
        X_train.append(augmented_image)
        y_train.append(augmented_steering)  


def load_data_set(root, file_name):
    """
    Loads the data set for all three cameras
    """
    print('>> Loading data set...')
    data_set_log  = load_data_set_log(root + '/' + file_name, dropping = True)
    
    X_train, y_train = [], []
    for idx in range(len(data_set_log)):
    #for idx in range(100):
        _probability = random.uniform(0.0, 1.0)
        
        if(_probability >= 0.00 and _probability < 0.50):# Images for center camera
            load_camera_data_set(
                X_train, y_train, 
                root + '/' + data_set_log.center[idx].strip(), 
                data_set_log.steering[idx],
                camera='c')
        elif(_probability >= 0.50 and _probability < 0.75):# Images for left camera
            load_camera_data_set(
                X_train, y_train, 
                root + '/' + data_set_log.left[idx].strip(), 
                data_set_log.steering[idx],
                camera='l')
        elif(_probability >= 0.75):# Images for right camera
            load_camera_data_set(
                X_train, y_train, 
                root + '/' + data_set_log.right[idx].strip(), 
                data_set_log.steering[idx],
                camera='r')
    
    X_train, y_train=np.array(X_train), np.array(y_train)
    X_train, y_train=shuffle(X_train, y_train)
    
    print('>> Done...')
    
    return X_train, y_train


def normalize_image(img):
    """
    Normalizes an image
    """
    return (img / 255.0) - 0.5


def pre_processing_model(input_shape=(160, 320, 3), cropping=((50,20), (0,0))):
    """
    Adds two layers to the model. 
    The first for normalaizing the input and the second layer for cropping the region of interest.
    """
    
    model = Sequential()
    model.add(Cropping2D(cropping=cropping, input_shape=input_shape))

    _shape = (input_shape[0] - sum(cropping[0]), 
              input_shape[1] - sum(cropping[1]), 
              input_shape[2])
    model.add(Lambda(normalize_image, input_shape=_shape, output_shape=_shape))
    
    return model


def lenet_model(input_shape=(160, 320, 3), drop_out = 0.5, drop_out_sp = 0.2):
    """
    LeNet Architecture
    """
    print('\n\n ')
    print('>> Building the model (LeNet Architecture)...')
    
    # Pre-processing layer
    model = pre_processing_model(input_shape=input_shape)
    
    # Other layers
    model.add(Convolution2D(6, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(SpatialDropout2D(drop_out_sp))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Convolution2D(6, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(SpatialDropout2D(drop_out_sp))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dropout(drop_out))
    model.add(Dense(120))
    model.add(Dropout(drop_out))
    model.add(Dense(84))
    model.add(Dropout(drop_out))
    model.add(Dense(1))
    
    model.summary()
    return model


def nvidia_model(input_shape=(160, 320, 3), drop_out = 0.5, drop_out_sp = 0.2):
    """
    NVIDIA Architecture
    src:[http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf]
    """
    print('\n\n ')
    print('>> Building the model (NVIDIA Architecture)...')
    
    # Pre-processing layer
    model = pre_processing_model(input_shape=input_shape)
    
    # Other layers
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(SpatialDropout2D(drop_out_sp))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(SpatialDropout2D(drop_out_sp))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(SpatialDropout2D(drop_out_sp))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(SpatialDropout2D(drop_out_sp))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(drop_out))
    model.add(Dense(100))
    model.add(Dropout(drop_out))
    model.add(Dense(50))
    model.add(Dropout(drop_out))
    model.add(Dense(10))
    model.add(Dropout(drop_out))
    model.add(Dense(1))
    
    model.summary()
    return model


def get_optimaizer(lr=0.001):
    """
    Returns an optimizer to reduce the loss.
    """
    return Adam(lr=lr)


def train_model(model, model_name, X_train, y_train, loss='mse', lr=0.001, valid_split=0.2, epochs=7, batch_size=32):
    """
    Trains the model
    """
    print('>> Training the model...')
    _optimizer = get_optimaizer(lr)
    print('>> Info: loss={0}, optimizer={1}, valid_split={2:.2f}, epochs={3}'
          .format(loss, _optimizer, valid_split, epochs))
    
    model.compile(loss=loss, optimizer=_optimizer)

    history_object = model.fit(
        X_train, y_train, validation_split=valid_split, 
        shuffle=True, epochs = epochs, batch_size=batch_size)
    
    model.save(model_name + '.h5')
    print('>> Model saved!')
    
    return history_object
    
    
def produce_visualization(history_object, model_name):
    """
    Produce the MSE Loss vs diagram
    """
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Mean squared error loss (MSEL) for "{0}"'.format(model_name))
    plt.ylabel('MSEL')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig('./res/msel_vs_epoch_{0}.png'.format(model_name), dpi=300)


def plot_data_dist(steering, bins=25, labelx ='data', save=False, save_path = None):
    """
    Plots the steering data set distribution
    """
    print(type(steering))
    steering.plot.hist(bins=bins)
    plt.xlabel(labelx)
    plt.grid('on')
    
    fig = plt.gcf()
    plt.show()
    plt.draw()
    
    if(save):
        if(save_path is None): raise Exception('"save-pathe" is None!')
        fig.savefig(save_path, dpi=300)