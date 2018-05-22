import numpy as np
import sys
from lycon import resize, Interpolation
from scipy.ndimage import gaussian_filter

def resize_batch(data, dataShape, type="bilinear"):
	# No need to do anything if shapes are identical.
	if data.shape[1 : ] == dataShape:
		return np.copy(data)

	assert type in ("bilinear", "nearest")
	numData = len(data)
	newData = np.zeros((numData, *dataShape), dtype=data.dtype)

	interpolationType = Interpolation.LINEAR if type == "bilinear" else Interpolation.NEAREST
	for i in range(len(data)):
		result = resize(data[i], height=dataShape[0], width=dataShape[1], interpolation=interpolationType)
		newData[i] = result.reshape(newData[i].shape)
	return newData

def standardizeData(data, mean, std):
	data = np.float32(data)
	data -= mean
	data /= std
	return data

def minMaxNormalizeData(data, min, max):
	data = np.float32(data)
	data -= min
	data /= (max - min)
	return data

def toCategorical(data, numClasses):
	numData = len(data)
	newData = np.zeros((numData, numClasses), dtype=np.uint8)
	newData[np.arange(numData), data] = 1
	return newData

# Labels can be None, in that case only data is available (testing cases without labels)
def makeGenerator(data, labels, batchSize):
	while True:
		numData = data.shape[0]
		numIterations = numData // batchSize + (numData % batchSize != 0)
		for i in range(numIterations):
			startIndex = i * batchSize
			endIndex = np.minimum((i + 1) * batchSize, numData)
			if not labels is None:
				yield data[startIndex : endIndex], labels[startIndex : endIndex]
			else:
				yield data[startIndex : endIndex]

def NoneAssert(conndition, noneCheck, message=""):
	if noneCheck:
		assert conndition, message

class LinePrinter:
	def __init__(self):
		self.maxLength = 0

	def print(self, message):
		if message[-1] == "\n":
			message = message[0 : -1]
			additional = "\n"
		else:
			additional = "\r"

		self.maxLength = np.maximum(len(message), self.maxLength)
		message += (self.maxLength - len(message)) * " " + additional
		sys.stdout.write(message)
		sys.stdout.flush()

# @brief Returns true if whatType is subclass of baseType. The parameters can be instantiated objects or types. In the
#  first case, the parameters are converted to their type and then the check is done.
def isBaseOf(whatType, baseType):
	if type(whatType) != type:
		whatType = type(whatType)
	if type(baseType) != type:
		baseType = type(baseType)
	return baseType in type(object).mro(whatType)