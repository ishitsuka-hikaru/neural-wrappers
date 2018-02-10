import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

def gaussian_downsample(image, output_shape):
	assert len(image.shape) == len(output_shape), "Expected same shape for both input and output_shape"
	factors = (np.asarray(image.shape, dtype=float) / np.asarray(output_shape, dtype=float))
	anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
	return gaussian_filter(image, anti_aliasing_sigma, cval=0, mode="reflect")

def anti_alias_resize(image, output_shape):
	output_shape = tuple(output_shape)
	# (480, 640, 3) and (240, 320) => (480, 640, 3) and (240, 320, 3)
	if len(output_shape) == len(image.shape) - 1:
		output_shape += (image.shape[-1], )

	# No need to do anything if shapes are identical.
	if output_shape == image.shape:
		return image

	image = gaussian_downsample(image, output_shape)
	return resize(image, output_shape, mode="reflect")

def anti_alias_resize_batch(data, dataShape):
	numData = len(data)
	newData = np.zeros((numData, *dataShape), dtype=data.dtype)
	# TODO: find a way to apply the resize on a batch
	for i in range(len(data)):
		newData[i] = anti_alias_resize(data[i], dataShape)
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
