import sys
sys.path.insert(0, '/home/xiaotian/caffe/python')
import caffe

import numpy as np 
import cv2
import copy
import img_process
import config

class DataLayerTrain(caffe.Layer):
	def setup(self, bottom, top):
		self._batch_size = config.BATCH_SIZE 
		self._class_num = config.CLASS_NUM 
		self._image_width = config.IMAGE_WIDTH 
		self._image_hight = config.IMAGE_HIGHT 
		self._index = 0

		file = config.TRAIN_FILE 

		self._data = []
		with open(file) as f:
			for line in f.readlines():
				line = line.strip()
				arr = line.split(',')

				l = np.zeros(self._class_num, dtype=float)
				for i in xrange(1, len(arr), 1):
					index = int(arr[i])
					l[index] = 1.0

				self._data.append([arr[0], l])

		np.random.shuffle(self._data)

		top[0].reshape(self._batch_size, 3, self._image_hight, self._image_width)
		top[1].reshape(self._batch_size, self._class_num)

	def forward(self, bottom, top):
		imgs_blob = []
		labels_blob = []

		if (self._index + self._batch_size > len(self._data)):
			np.random.shuffle(self._data)
			self._index = 0

		while (len(imgs_blob) < self._batch_size):
			img = cv2.imread(self._data[self._index][0])
			label = self._data[self._index][1]

			if (img is None) or (len(img) < 32) or (len(img[0]) < 32):
				self._index += 1
				continue

			randint = np.random.randint(2)
			if randint == 1:
				img = img[:,::-1,:]

			imgs_blob.append(img_process.img_to_blob(img, self._image_hight, self._image_width))
			labels_blob.append(label)

			self._index += 1

		top[0].data[...] = np.array(imgs_blob)
		top[1].data[...] = np.array(labels_blob)

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass


class DataLayerTest(caffe.Layer):
	def setup(self, bottom, top):
		self._batch_size = config.BATCH_SIZE 
		self._class_num = config.CLASS_NUM 
		self._image_width = config.IMAGE_WIDTH 
		self._image_hight = config.IMAGE_HIGHT 
		self._index = 0

		file = config.TEST_FILE

		self._data = []
		with open(file) as f:
			for line in f.readlines():
				line = line.strip()
				arr = line.split(',')

				l = np.zeros(self._class_num, dtype=float)
				for i in xrange(1, len(arr), 1):
					index = int(arr[i])
					l[index] = 1.0

				self._data.append([arr[0], l])

		np.random.shuffle(self._data)

		top[0].reshape(self._batch_size, 3, self._image_hight, self._image_width)
		top[1].reshape(self._batch_size, self._class_num)

	def forward(self, bottom, top):
		imgs_blob = []
		labels_blob = []

		if (self._index + self._batch_size > len(self._data)):
			np.random.shuffle(self._data)
			self._index = 0

		while (len(imgs_blob) < self._batch_size):
			img = cv2.imread(self._data[self._index][0])
			label = self._data[self._index][1]

			if (img is None) or (len(img) < 32) or (len(img[0]) < 32):
				self._index += 1
				continue

			randint = np.random.randint(2)
			if randint == 1:
				img = img[:,::-1,:]

			imgs_blob.append(img_process.img_to_blob(img, self._image_hight, self._image_width))
			labels_blob.append(label)

			self._index += 1

		top[0].data[...] = np.array(imgs_blob)
		top[1].data[...] = np.array(labels_blob)

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass