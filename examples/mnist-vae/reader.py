import numpy as np
from neural_wrappers.readers import MNISTReader

class BinaryMNISTReader(MNISTReader):
	def __getitem__(self, index):
		item = super().__getitem__(index)
		images = item["data"]["images"]
		images = np.float32(images > 0)
		return images, images
