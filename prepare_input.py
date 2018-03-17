import time
import os, os.path
import numpy as np
import cv2
from keras.utils import np_utils
from scipy import misc

images_path = os.path.join(os.path.dirname(__file__), 'images2')

def prepare_input():
		start = time.time()
		data = []
		label = []
		folder_no = 0
		for folder in os.listdir(images_path):
			class_images_path = images_path+'/'+folder
			print('Moving to: '+ class_images_path)
			num_example = len([image for image in os.listdir(class_images_path)])
			for image in os.listdir(class_images_path):
				print('Creating array for'+class_images_path+'/'+image+'...')
				image_array = cv2.imread(class_images_path+'/'+image)
				#data.append(image_array)
				resized_image_array = cv2.resize(image_array, (299, 299))
				data.append(resized_image_array)
				label.append(folder_no)
			folder_no+=1
		data = np.array(data)
		label = np.array(label)
		
		p = np.random.permutation(data.shape[0])
		X = data[p]
		y = label[p]
		X = X.astype('float32')
		X = X/ 255.0	
		
		ratio = round(0.9 * data.shape[0])
		X_train = X[:int(ratio)]
		y_train = y[:int(ratio)]
		X_test = X[int(ratio):]
		y_test = y[int(ratio):]
		
		y_train_vector = np_utils.to_categorical(y_train)
		y_test_vector = np_utils.to_categorical(y_test)

		#print(X_train.shape, X_test.shape)
		#print(y_train[:10], y_test[:10])
		print("Time taken for creating numpy arrays: ", time.time() - start)
		
		np.save('./numpy_arrays/X_train_images2_inception.npy', X_train)
		np.save('./numpy_arrays/y_train_images2_inception.npy', y_train_vector)
		np.save('./numpy_arrays/X_test_images2_inception.npy', X_test)
		np.save('./numpy_arrays/y_test_images2_inception.npy', y_test_vector)
		
if __name__ == '__main__':
	prepare_input()