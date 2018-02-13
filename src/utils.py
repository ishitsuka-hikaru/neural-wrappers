import numpy as np
from lycon import resize, Interpolation
from scipy.ndimage import gaussian_filter

def anti_alias_resize_batch(data, dataShape):
	# No need to do anything if shapes are identical.
	if data.shape == dataShape:
		return data

	numData = len(data)
	newData = np.zeros((numData, *dataShape), dtype=data.dtype)

	for i in range(len(data)):
		newData[i] = resize(data[i], height=dataShape[0], width=dataShape[1], interpolation=Interpolation.NEAREST)
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
