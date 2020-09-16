import numpy as np
from neural_wrappers.readers import MNISTReader

class BinaryMNISTReader(MNISTReader):
	def iterateOneEpoch(self, type, miniBatchSize):
		for items in super().iterateOneEpoch(type, miniBatchSize):
			images, _ = items
			images = np.float32(images > 0)
			yield images, images
