import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def some_function_name(data):
	# batch_size = 128
	# num_classes = 10
	# epochs = 1
	#
	# num_training_examples = data['x_train'].shape[0]
	# num_features = 784
	# data['x_train'] = data['x_train'].reshape(num_training_examples, num_features)
	# data['y_train'] = data['y_train'].reshape(num_training_examples, 1)
	# data['x_train'] = data['x_train'].astype('float32')
	# data['x_train'] /= 255
	# data['y_train'] = keras.utils.to_categorical(data['y_train'], num_classes)
	#
	# model = Sequential()
	# model.add(Dense(512, activation='relu', input_shape=(num_features,)))
	# model.add(Dropout(0.2))
	# model.add(Dense(512, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(num_classes, activation='softmax'))
	#
	# model.compile(loss='categorical_crossentropy',
	# 	optimizer=Adam(),
	# 	metrics=['accuracy'])

	result = data

	# history = model.fit(data['x_train'], data['y_train'],
	# 	batch_size=batch_size,
	# 	epochs=epochs,
	# 	verbose=1)

	return result
