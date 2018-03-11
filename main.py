import numpy as np
import cv2
import os, os.path

import matplotlib.pyplot as plt
from scipy.misc import toimage

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

class Object_identification(object):
	def __init__(self):
		self.images_path = os.path.join(os.path.dirname(__file__), 'images')
		self.videos_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_360P')
		self.num_example = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None	
		self.y_train_vector = None
		self.y_test_vector = None
		self.num_labels = 2
		self.output_dim = 2
		self.class_var = [0,1]
		self.model = None
		self.epochs = 2
		self.seed = 7
	
	def split_video_into_frames():
		try:
			if not os.path.exists('images'):
				os.makedirs('images')
		except OSError:
			print ('Error: Creating directory of images')
		currentFrame = 0
		for video in os.listdir(self.videos_path):
			cap = cv2.VideoCapture(self.videos_path+'/'+video)			
			for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
				ret, frame = cap.read()
				name = './images/frame' + str(currentFrame) + '.jpg'
				print ('Creating...' + name)
				cv2.imwrite(name, frame)
				currentFrame += 1
			cap.release()
			cv2.destroyAllWindows()
		
	def prepare_input(self):
		print('preparing input...')
		data = []
		label = []
		self.num_example = len([name for name in os.listdir(self.images_path)])
		
		for i in range(1000,1100):
			image_array = cv2.imread(self.images_path+'/frame'+str(i)+'.jpg')
			#print(image_array.shape)
			data.append(image_array)
			label.append(0)
		print('next...')
		for i in range(23000,23100):
			image_array = cv2.imread(self.images_path+'/frame'+str(i)+'.jpg')
			#print(image_array.shape)
			data.append(image_array)
			label.append(1)
		data = np.array(data)
		label = np.array(label)
		print(data.shape)
		
		p = np.random.permutation(data.shape[0])
		X = data[p]
		y = label[p]
		X = X.astype('float32')
		X = X/ 255.0	
		
		ratio = round(0.9 * data.shape[0])
		self.X_train = X[:int(ratio)]
		self.y_train = y[:int(ratio)]
		self.X_test = X[int(ratio):]
		self.y_test = y[int(ratio):]
		
		self.y_train_vector = np_utils.to_categorical(self.y_train)
		self.y_test_vector = np_utils.to_categorical(self.y_test)

		print(self.X_train.shape, self.X_test.shape)
		print(self.y_train.shape, self.y_test.shape)

	def plot_images(self):
		for i in range(0, 9):
			plt.subplot(330 + 1 + i)
			plt.imshow(toimage(self.X_train[i]))	
		plt.show()
		
	def define_model(self):
		np.random.seed(self.seed)
		print('defining model...')
		model = Sequential()
		model.add(Conv2D(4, (6, 6), input_shape=(360, 640, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
		#model.add(Dropout(0.2))
		#model.add(Conv2D(3, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(10, activation='relu', kernel_constraint=maxnorm(3)))
		model.add(Dropout(0.5))
		model.add(Dense(self.num_labels, activation='softmax'))
		lrate = 0.01
		decay = lrate/self.epochs
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		print(model.summary())
		self.model = model
		
	def learn_model(self):
		print('learning model...')
		self.model.fit(self.X_train, self.y_train_vector, validation_data=(self.X_test, self.y_test_vector), epochs=self.epochs, batch_size=32)
		scores = self.model.evaluate(self.X_test, self.y_test_vector, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
	obj = Object_identification()
	#obj.split_video_into_frames()
	obj.prepare_input()
	#obj.plot_images()
	obj.define_model()
	obj.learn_model()
