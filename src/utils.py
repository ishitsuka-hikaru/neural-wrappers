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
		newData[i] = resize(data[i], height=dataShape[0], width=dataShape[1], interpolation=interpolationType)
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

class LinePrinter:
	def __init__(self):
		self.maxLength = 0

	def print(self, message):
		self.maxLength = np.maximum(len(message), self.maxLength)
		message += (self.maxLength - len(message)) * " "
		sys.stdout.write(message + "\r")
		sys.stdout.flush()
