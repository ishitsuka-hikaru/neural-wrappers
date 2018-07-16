import numpy as np

# Generic transform
class Transform:
	# Data is assumed batched (MB x *dataShape)
	def __call__(self, data):
		raise NotImplementedError("Should have implemented this")

	def __str__(self):
		return "Generic data transformation"

# Image mirroring transform, only valid for image inputs, batched in 2D (grayscale) or 3D (RGB, HSV etc.) formats.
class Mirror(Transform):
	def __call__(self, data):
		return np.flip(data, 2)

	def __str__(self):
		return "Image mirroring transformation"

class CropMiddle(Transform):
	def __init__(self, desiredShape):
		self.desiredShape = desiredShape

	def __call__(self, data):
		assert data.shape[1] > self.desiredShape[0] and data.shape[2] > self.desiredShape[1]
		indexes = self.computeIndexes(data.shape)
		return data[:, indexes[0] : -indexes[1], indexes[2] : -indexes[3]]

	# Example: 28x28 image, 20x20 desiredShape => indexes are 4 and 24
	def computeIndexes(self, dataShape):
		diffTop = (dataShape[1] - self.desiredShape[0]) // 2 + ((dataShape[1] - self.desiredShape[0]) % 2 == 1)
		diffBottom = (dataShape[1] - self.desiredShape[0]) // 2
		diffLeft = (dataShape[2] - self.desiredShape[1]) // 2 + ((dataShape[2] - self.desiredShape[1]) % 2 == 1)
		diffRight = (dataShape[2] - self.desiredShape[1]) // 2
		return diffTop, diffBottom, diffLeft, diffRight

	def __str__(self):
		return "Crop middle transformation"

class CropTopLeft(Transform):
	def __init__(self, desiredShape):
		self.desiredShape = desiredShape

	def __call__(self, data):
		assert data.shape[1] > self.desiredShape[0] and data.shape[2] > self.desiredShape[1]
		return data[:, 0 : self.desiredShape[0], 0 : self.desiredShape[1]]

	def __str__(self):
		return "Crop top left transformation"

class CropTopRight(Transform):
	def __init__(self, desiredShape):
		self.desiredShape = desiredShape

	def __call__(self, data):
		assert data.shape[1] > self.desiredShape[0] and data.shape[2] > self.desiredShape[1]
		return data[:, 0 : self.desiredShape[0], -self.desiredShape[1] : ]

	def __str__(self):
		return "Crop top right transformation"

class CropBottomLeft(Transform):
	def __init__(self, desiredShape):
		self.desiredShape = desiredShape

	def __call__(self, data):
		assert data.shape[1] > self.desiredShape[0] and data.shape[2] > self.desiredShape[1]
		return data[:, -self.desiredShape[0] : , 0 : self.desiredShape[1]]

	def __str__(self):
		return "Crop bottom left transformation"

class CropBottomRight(Transform):
	def __init__(self, desiredShape):
		self.desiredShape = desiredShape

	def __call__(self, data):
		assert data.shape[1] > self.desiredShape[0] and data.shape[2] > self.desiredShape[1]
		return data[:, -self.desiredShape[0] : , -self.desiredShape[1] : ]

	def __str__(self):
		return "Crop bottom right transformation"
