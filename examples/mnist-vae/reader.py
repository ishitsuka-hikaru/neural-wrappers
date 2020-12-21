import numpy as np
from neural_wrappers.readers import MNISTReader

class BinaryMNISTReader(MNISTReader):
	def getBatchItem(self, index):
		item = super().getBatchItem(index)
		images = item["data"]["images"]
		images = np.float32(images > 0)
		return images, images
