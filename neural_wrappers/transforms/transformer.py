# transformer.py Generic class for data augmentation. Definitely not Optimus Prime.
from neural_wrappers.utilities import resize_batch, identity, isSubsetOf
from .transforms import Mirror
import numpy as np

class Transformer:
	# @param[in] allDims A list of dimensions that are required for this transformer. The transformations done to each
	#  of them (even if it's just identity).
	# @param[in] transforms A list of transformations, that must be applied, when calling the applyTransforms method.
	#  Each element of the list must be a dictionary, with keys from the allDims parameter (or otherwise defaulted to
	#  identity).
	def __init__(self, allDims, transforms):
		assert type(transforms) == list
		self.allDims = allDims
		self.transforms = []
		self.transformNames = []
		self.builtInTransforms = self.getBuiltInTransforms()

		if len(transforms) == 0:
			transforms = [{dim : "identity" for dim in self.allDims}]

		for i in range(len(transforms)):
			assert type(transforms[i]) == dict
			assert isSubsetOf(list(transforms[i].keys()), allDims)

			# Update the functions (if using built-in) and store the transform
			updatedTransforms = self.updateTransforms(transforms[i])
			self.transforms.append(updatedTransforms)

			# Check for duplicates as well (all transforms must be unique, even the names)
			transformName = {dim : updatedTransforms[dim][0] for dim in self.allDims}
			assert not transformName in self.transformNames, \
				"%s already exist for this transformer (same names)" % (transformName)
			self.transformNames.append(transformName)

	def getBuiltInTransforms(self):
		mirror = Mirror()
		builtInTransforms = {
			"none" : identity,
			"identity" : identity,
			"mirror" : Mirror(),
		}

		return builtInTransforms

	def updateTransforms(self, transforms):
		updateTransforms = {}
		for key in self.allDims:
			# If the user didn't specify one or more dimensions, these are defaulted to identity
			if not key in transforms:
				updateTransforms[key] = ("identity", identity)
				continue

			transform = transforms[key]
			if transform in self.builtInTransforms:
				updateTransforms[key] = (transform, self.builtInTransforms[transform])
			else:
				assert type(transform) == tuple
				assert type(transform[0]) == str and hasattr(transform[1], "__call__"), \
				("The user provided transform \"%s\" must be a tuple of type (Str, Callable) or must one of the " + \
				"default transforms") % transform[0]
				updateTransforms[key] = transform
		return updateTransforms

	# Main function, that loops through all transforms and through all data (and labels) and yields each transformed
	#  version for the main caller.
	# @param[in] data The original data, unaltered in any way, on which the transforms (and potential resizes) are done
	# @param[in] labels (optional) The original labels (or list of labels) on which the transforms are done, which are
	#  done in same manner as they are done for the data (for example for random cropping, same random indexes are
	#  chosen).
	def applyTransforms(self, data):
		for transform in self.transforms:
			newData = {}
			for dim in data:
				transformName, transformFunc = transform[dim]
				# print("Applying %s for %s" % (transformName, dim))
				newData[dim] = transformFunc(np.copy(data[dim]))
			yield newData

	def __call__(self, data):
		return self.applyTransforms(data)

	def __str__(self):
		Str = "[Transformer]\n"
		for transform in self.transforms:
			transformName = {dim : transform[dim][0] for dim in self.allDims}
			Str += "  - %s\n" % (transformName)
		return Str