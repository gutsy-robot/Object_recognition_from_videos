import time
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import cv2

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
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input

class Object_identification(object):
	def __init__(self):
		self.images_path = os.path.join(os.path.dirname(__file__), 'images')
		self.videos_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_360P')
		self.results_path = os.path.join(os.path.dirname(__file__), 'results')
		self.num_example = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.y_predict = None
		self.y_train_vector = None
		self.y_test_vector = None
		self.y_predict_vector = None
		self.num_labels = 2
		self.output_dim = 2
		self.class_var = [0,1]
		self.model = None
		self.history = None
		self.epochs = 2
		self.seed = 7
		self.w = 1
	
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
		start = time.time()
		print('preparing input...')
		data = []
		label = []
		self.num_example = len([name for name in os.listdir(self.images_path)])
		
		for i in range(1000,1100):
			image_array = cv2.imread(self.images_path+'/frame'+str(i)+'.jpg')
			#data.append(image_array)
			resized_image_array = cv2.resize(image_array, (299, 299))
			data.append(resized_image_array)
			label.append(0)
		print('next...')
		for i in range(23000,23100):
			image_array = cv2.imread(self.images_path+'/frame'+str(i)+'.jpg')
			#data.append(image_array)
			resized_image_array = cv2.resize(image_array, (299, 299))
			data.append(resized_image_array)
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
		#print(self.y_train[:10], self.y_test[:10])
		print("Time taken for creating numpy arrays: ", time.time() - start)

	def plot_images(self, X, y):
		for i in range(0, 20):
			plt.subplot(4,5,1 + i)	
			plt.imshow(toimage(X[i]))
			plt.title('label: '+str(y[i]))
		plt.show()
		
	def plot_loss(self):
		plt.plot(self.history.history['loss'], color='red')
		plt.plot(self.history.history['val_loss'], color='blue')
		plt.xlabel('Epochs')
		plt.ylabel('Error')
		plt.legend(['Train', 'Validation'], loc='upper right')
		plt.title('  l:'+'%.4f'%self.history.history['loss'][-1]+'  a:'+'%.2f'%(self.history.history['acc'][-1]*100)+'  val_l:'+'%.4f'%self.history.history['val_loss'][-1]+
		'  val_a:'+'%.2f'%(self.history.history['val_acc'][-1]*100))
		plt.savefig(str(self.results_path)+'/graph'+str(self.w)+'.png')
		plt.close()
		
	def define_model(self):
		np.random.seed(self.seed)
		print('defining model...')
		model = Sequential()
		model.add(Conv2D(4, (3, 3), input_shape=(360, 640, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Conv2D(4, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Conv2D(4, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		#model.add(Conv2D(4, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
		#model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		#model.add(Dense(10, activation='relu', kernel_constraint=maxnorm(3)))
		#model.add(Dropout(0.5))
		model.add(Dense(self.num_labels, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		print(model.summary())
		self.model = model
		
	def learn_model(self):
		start = time.time()
		print('learning model...')		
		self.history = self.model.fit(self.X_train, self.y_train_vector, validation_data=(self.X_test, self.y_test_vector), epochs=self.epochs, batch_size=512)
		print("training Time : ", time.time() - start)
		scores = self.model.evaluate(self.X_test, self.y_test_vector, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))
		self.y_predict_vector = self.model.predict(self.X_test)
		self.y_predict = np.argmax(self.y_predict_vector, axis=1)
		print(self.y_predict[:10], self.y_test_vector[:10])
		
	def save_model(self):
		try:
			if not os.path.exists('results'):
				os.makedirs('results')
		except OSError:
			print ('Error: Creating results folder')
		self.model.save('./results/model'+str(self.w)+'.h5')
		
	def using_saved_model(self):
		self.model = load_model('./results/model'+str(self.w)+'.h5')
		scores = self.model.evaluate(self.X_test, self.y_test_vector, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))
		self.y_predict_vector = self.model.predict(self.X_test)
		self.y_predict = np.argmax(self.y_predict_vector, axis=1)
		
	def using_pretrained_model(self):
		start = time.time()
		base_model = InceptionV3(include_top=False, weights='imagenet')
		#base_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		features_train = base_model.predict(self.X_train, verbose=0)
		features_test = base_model.predict(self.X_test, verbose=0)
		print('Time taken by pretrained model:', time.time() - start)
		print(features_train.shape, features_test.shape)
		
		model = Sequential()
		model.add(Flatten(input_shape=(8, 8, 2048)))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(self.num_labels, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		print(model.summary())
		self.model = model
		self.history = self.model.fit(features_train, self.y_train_vector, validation_data=(features_test, self.y_test_vector), epochs=self.epochs, batch_size=512)
		scores = self.model.evaluate(features_test, self.y_test_vector, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))
		self.y_predict_vector = self.model.predict(features_test)
		self.y_predict = np.argmax(self.y_predict_vector, axis=1)
		
if __name__ == '__main__':
	obj = Object_identification()
	#obj.split_video_into_frames()
	obj.prepare_input()
	obj.using_pretrained_model()
	obj.plot_images(obj.X_test, obj.y_predict)
	'''
	obj.use_saved_model = True
	if obj.use_saved_model:
		obj.using_saved_model()
		obj.plot_images(obj.X_test, obj.y_predict)
	else:
		obj.define_model()
		obj.learn_model()
		obj.save_model()
		obj.plot_images(obj.X_test, obj.y_predict)
		obj.plot_loss()
	'''
