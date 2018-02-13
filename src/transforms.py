import numpy as np
from utils import anti_alias_resize_batch

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
		self.transforms = transforms
		self.dataShape = dataShape
		self.labelShape = labelShape

		# This parameter controls if the transformation for the labels is done on the shape of the data or the label
		sentLabelShape = self.dataShape if applyOnDataShapeForLabels else self.labelShape

		# Built-in transforms
		mirror = Mirror(dataShape, sentLabelShape)
		cropTopLeft = CropTopLeft(dataShape, sentLabelShape)
		cropTopRight = CropTopRight(dataShape, sentLabelShape)
		cropBottomLeft = CropBottomLeft(dataShape, sentLabelShape)
		cropBottomRight = CropBottomRight(dataShape, sentLabelShape)

		# There are some built-in transforms that can be sent as strings. For more complex ones, a lambda functon
		#  or a class that implements the __call__ function must be used. See class Transform for parameters.
		for i, transform in enumerate(self.transforms):
			if transform == "mirror":
				self.transforms[i] = mirror
			elif transform == "crop_top_left":
				self.transforms[i] = cropTopLeft
			elif transform == "crop_top_right":
				self.transforms[i] = cropTopRight
			elif transform == "crop_bottom_left":
				self.transforms[i] = cropBottomLeft
			elif transform == "crop_bottom_right":
				self.transforms[i] = cropBottomRight
			elif transform == "crop_top_left_mirror":
				self.transforms[i] = lambda data, labels: mirror(*cropTopLeft(data, labels))
			elif transform == "crop_top_right_mirror":
				self.transforms[i] = lambda data, labels: mirror(*cropTopRight(data, labels))
			elif transform == "crop_bottom_left_mirror":
				self.transforms[i] = lambda data, labels: mirror(*cropBottomLeft(data, labels))
			elif transform == "crop_bottom_right_mirror":
				self.transforms[i] = lambda data, labels: mirror(*cropBottomRight(data, labels))
			elif transform == "none":
				self.transforms[i] = lambda data, labels: (data, labels)
			else:
				assert hasattr(transform, "__call__"), "The user provided transformation %s must be callable" % \
					(transform)

	# TODO: make it generic, so it can receive or not a label (or a list of labels) and apply the transforms on them
	#  identically. Example of usage: random crop on both depth and semantic segmentation image, but we want to have
	#  an identical random crop for all 3 images (rgb, depth and segmentation).
	def applyTransform(self, transform, data, labels):
		print("Applying '%s' transform" % (transform))
		numData = len(data)
		if transform == "none":
			transform = lambda data, labels: (data, labels)
		newData, newLabels = transform(data, labels)

		newData = anti_alias_resize_batch(newData, self.dataShape)
		newLabels = anti_alias_resize_batch(newLabels, self.labelShape)

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
	def applyTransforms(self, data, labels=None):
		# assert self.labelsPresent == False or (self.labelsPresent == True and not labels is None)
		for transform in self.transforms:
			yield self.applyTransform(transform, data, labels)
