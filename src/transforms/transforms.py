import numpy as np
from utils import NoneAssert

# Generic transform
class Transform:
	def __init__(self, dataShape, labelsShape):
		self.dataShape = dataShape
		self.labelsShape = labelsShape

	# Main function that is called to apply a transformation on one data item
	# TODO: make label a list or None (case explained below with depth+semantic)
	def __call__(self, data, labels):
		raise NotImplementedError("Should have implemented this")

	def __str__(self):
		return "Generic data transformation"

class Mirror(Transform):
	def __call__(self, data, labels):
		# Expected NxHxWxD data or NxHxW. For anything else, implement your own mirroring.
		assert len(data.shape) in (3, 4)
		if not labels is None: assert len(labels.shape) in (3, 4)

		newData = np.flip(data, 2)
		newLabels = np.flip(labels, 2) if not labels is None else None
		return newData, newLabels

	def __str__(self):
		return "Image mirroring transformation"

class CropMiddle(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		dataIndexes = self.computeIndexes(data.shape)
		newData = data[:, dataIndexes[0] : -dataIndexes[1], dataIndexes[2] : -dataIndexes[3]]
		newLabels = None

		if not labels is None:
			labelIndexes = self.computeIndexes(labels.shape)
			newLabels = labels[:, labelIndexes[0] : -labelIndexes[1], labelIndexes[2] : -labelIndexes[3]]

		return newData, newLabels

	def computeIndexes(self, dataShape):
		diffTop = (dataShape[1] - self.dataShape[0]) // 2 + ((dataShape[1] - self.dataShape[0]) % 2 == 1)
		diffBottom = (dataShape[1] - self.dataShape[0]) // 2
		diffLeft = (dataShape[2] - self.dataShape[1]) // 2 + ((dataShape[2] - self.dataShape[1]) % 2 == 1)
		diffRight = (dataShape[2] - self.dataShape[1]) // 2
		return diffTop, diffBottom, diffLeft, diffRight

	def __str__(self):
		return "Crop middle transformation"

class CropTopLeft(Transform):
	def __call__(self, data, labels):
		# Remember that first dimension is the batch
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, 0 : self.dataShape[0], 0 : self.dataShape[1]]
		newLabels = None

		if not labels is None:
			newLabels = labels[:, 0 : self.labelsShape[0], 0 : self.labelsShape[1]]
		return newData, newLabels

	def __str__(self):
		return "Crop top left transformation"

class CropTopRight(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, 0 : self.dataShape[0], -self.dataShape[1] : ]
		newLabels = None
		if not labels is None:
			newLabels = labels[:, 0 : self.labelsShape[0], -self.labelsShape[1] : ]
		return newData, newLabels

	def __str__(self):
		return "Crop top right transformation"

class CropBottomLeft(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, -self.dataShape[0] : , 0 : self.dataShape[1]]
		newLabels = None
		if not labels is None:
			newLabels = labels[:, -self.labelsShape[0] : , 0 : self.labelsShape[1]]
		return newData, newLabels

	def __str__(self):
		return "Crop bottom left transformation"

class CropBottomRight(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, -self.dataShape[0] : , -self.dataShape[1] : ]
		newLabels = None
		if not labels is None:
			newLabels = labels[:, -self.labelsShape[0] : , -self.labelsShape[1] : ]
		return newData, newLabels

	def __str__(self):
		return "Crop bottom right transformation"
