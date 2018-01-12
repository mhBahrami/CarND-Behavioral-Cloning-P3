import helper

# The directory information to the data set
root = './data'
file_name = 'driving_log.csv'


# Load data set 
X_train, y_train = helper.load_data_set(root, file_name)

# Obtain the shape of input
input_shape = X_train[0].shape
print('>> Data set size: {0}'.format(len(y_train)))
print('>> Image shape: {0}'.format(input_shape))


# Build the models
idx = 1
model = \
    helper.lenet_model(input_shape=input_shape) if idx==0 \
    else helper.nvidia_model(input_shape=input_shape)
model_names = ['lenet_model', 'nvidia_model']
model_name = model_names[idx]
print('\n\n ')
print('+-------------------------------------------------+')
print('|                   {0:12s}                  |'.format(model_name))
print('+-------------------------------------------------+')


# Train and save the model
history_object = helper.train_model(
    model, model_name, X_train, y_train, 
    loss='mse', lr=0.001,
    valid_split=0.2, epochs = 10)


# Produce and save visualization    
helper.produce_visualization(history_object, model_name)