import numpy as np
from utils import resize_batch

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
		assert len(data.shape) in (3, 4) and len(labels.shape) in (3, 4)
		newData = np.flip(data, 2)
		newLabels = np.flip(labels, 2)
		return newData, newLabels

	def __str__(self):
		return "Image mirroring transformation"

class CropMiddle(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		dataIndexes = self.computeIndexes(data.shape)
		labelIndexes = self.computeIndexes(labels.shape)
		newData = data[:, dataIndexes[0] : -dataIndexes[1], dataIndexes[2] : -dataIndexes[3]]
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
		newLabels = labels[:, 0 : self.labelsShape[0], 0 : self.labelsShape[1]]
		return newData, newLabels

	def __str__(self):
		return "Crop top left transformation"

class CropTopRight(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, 0 : self.dataShape[0], -self.dataShape[1] : ]
		newLabels = labels[:, 0 : self.labelsShape[0], -self.labelsShape[1] : ]
		return newData, newLabels

	def __str__(self):
		return "Crop top right transformation"

class CropBottomLeft(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, -self.dataShape[0] : , 0 : self.dataShape[1]]
		newLabels = labels[:, -self.labelsShape[0] : , 0 : self.labelsShape[1]]
		return newData, newLabels

	def __str__(self):
		return "Crop bottom left transformation"

class CropBottomRight(Transform):
	def __call__(self, data, labels):
		assert data.shape[1] > self.dataShape[0] and data.shape[2] > self.dataShape[1]
		newData = data[:, -self.dataShape[0] : , -self.dataShape[1] : ]
		newLabels = labels[:, -self.labelsShape[0] : , -self.labelsShape[1] : ]
		return newData, newLabels

	def __str__(self):
		return "Crop bottom right transformation"

# Generic class for data augmentation. Definitely not Optimus Prime.
class Transformer:
	# TODO: find a better name for last parameter. It is used for the case where I want to apply some transformation
	#  for example a top-left crop, but the labelShape (at the end) is smaller than the dataShape. If we were to apply
	#  the crop on the labelShape, we'd get a bad and different crop.
	#  Concrete case: image:(480x640x3), label:(480x640), dataShape:(240x320x3), desiredLabelShape:(50x70). The crop
	#  must be applied for values in [0 : 240, 0 : 320] on the (400x640) label, not for [0 : 50, 0 :70]. The reshape is
	#  done at end. There may also be cases where the crop must be done on [0 : 50, 0 : 70], so leave it as an option.
	def __init__(self, transforms, dataShape, labelShape, applyOnDataShapeForLabels=False):
		# assert labelsPresent == False or (labelsPresent == True and labelShape != None)
		self.dataShape = dataShape
		self.labelShape = labelShape
		self.applyOnDataShapeForLabels = applyOnDataShapeForLabels

		# This parameter controls if the transformation for the labels is done on the shape of the data or the label
		self.transforms = self.handleTransforms(transforms)

	def handleTransforms(self, transforms):
		sentLabelShape = self.dataShape if self.applyOnDataShapeForLabels else self.labelShape
		dictTransforms = {}
		dataShape = self.dataShape

		# Built-in transforms
		mirror = Mirror(dataShape, sentLabelShape)
		cropMiddle = CropMiddle(dataShape, sentLabelShape)
		cropTopLeft = CropTopLeft(dataShape, sentLabelShape)
		cropTopRight = CropTopRight(dataShape, sentLabelShape)
		cropBottomLeft = CropBottomLeft(dataShape, sentLabelShape)
		cropBottomRight = CropBottomRight(dataShape, sentLabelShape)
		builtInTransforms = {
			"none" : lambda data, labels: (data, labels),
			"mirror" : mirror,
			"crop_middle" : cropMiddle,
			"crop_top_left" : cropTopLeft,
			"crop_top_right" : cropTopRight,
			"crop_bottom_left" : cropBottomLeft,
			"crop_bottom_right" : cropBottomRight,
			"crop_middle_mirror" : lambda data, labels: mirror(*cropMiddle(data, labels)),
			"crop_top_left_mirror" : lambda data, labels: mirror(*cropTopLeft(data, labels)),
			"crop_top_right_mirror" : lambda data, labels: mirror(*cropTopRight(data, labels)),
			"crop_bottom_left_mirror" : lambda data, labels: mirror(*cropBottomLeft(data, labels)),
			"crop_bottom_right_mirror" : lambda data, labels: mirror(*cropBottomRight(data, labels))
		}

		if type(transforms) == list:
			# There are some built-in transforms that can be sent as strings. For more complex ones, a lambda functon
			#  or a class that implements the __call__ function must be used as well as a name, sent in a tuple/list.
			#  See class Transform for parameters for __call__. Example: ("cool_transform", lambda x, y : (x+1, y+1)).
			for i, transform in enumerate(transforms):
				if type(transform) == str:
					assert transform in builtInTransforms, "If only name is given, expect one of the built-in " + \
						"transforms to be given: %s" % (builtInTransforms.keys())
					dictTransforms[transform] = builtInTransforms[transform]
				elif type(transform) in (tuple, list):
					assert len(transform) == 2
					name, transformFunc = transform
					assert hasattr(transformFunc, "__call__"), "The user provided transformation %s must be " + \
						"callable" % (name)
					assert not name in builtInTransforms, "Cannot overwrite a built-in transform name: %s" % (name)
					assert not name in dictTransforms, "Cannot give the same name to two transforms: %s" % (name)
					dictTransforms[name] = transformFunc
				else:
					assert False, "Expected either a str for built-in or a (str, func) pair for user transform"
		elif type(transforms) == dict:
			dictTransforms = transforms
		else:
			raise Exception("Expected transforms to be a list or dict, got: " + type(transforms))
		return dictTransforms

	# TODO: make it generic, so it can receive or not a label (or a list of labels) and apply the transforms on them
	#  identically. Example of usage: random crop on both depth and semantic segmentation image, but we want to have
	#  an identical random crop for all 3 images (rgb, depth and segmentation).
	def applyTransform(self, transformName, data, labels, interpolationType):
		# print("Applying '%s' transform" % (transformName))
		numData = len(data)
		newData, newLabels = self.transforms[transformName](data, labels)

		newData = resize_batch(newData, self.dataShape, interpolationType)
		newLabels = resize_batch(newLabels, self.labelShape, interpolationType)

		assert newData.shape == (numData, *self.dataShape) and newData.dtype == data.dtype \
			and newLabels.shape == (numData, *self.labelShape) and newLabels.dtype == labels.dtype, "Expected data " \
			+ "shape %s, found %s. Expected labels shape %s, found %s." % (newData.shape, (numData, *self.dataShape),\
			newLabels.shape, (numData, *self.labelShape))
		return newData, newLabels

	# Main function, that loops throught all transforms and through all data (and labels) and yields each transformed
	#  version for the main caller.
	# @param[in] data The original data, unaltered in any way, on which the transforms (and potential resizes) are done
	# @param[in] labels (optional) The original labels (or list of labels) on which the transforms are done, which are
	#  done in same manner as they are done for the data (for example for random cropping, same random indexes are
	#  chosen).
	def applyTransforms(self, data, labels=None, interpolationType="bilinear"):
		# assert self.labelsPresent == False or (self.labelsPresent == True and not labels is None)
		for transform in self.transforms:
			yield self.applyTransform(transform, data, labels, interpolationType)
