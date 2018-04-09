import numpy as np
import h5py
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer
from neural_wrappers.utilities import standardizeData

# A secondary reader for CityScapes algorithm. This one instead of taking raw videos, expects a pre-compiuted h5py
#  file from which it loads the data.
class CityScapesImagesReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], flowAlgorithm=None):
		assert flowAlgorithm is None or flowAlgorithm in ("flownet2")
		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.transforms = transforms
		self.flowAlgorithm = flowAlgorithm

		self.dataAugmenter = Transformer(transforms, dataShape=imageShape, labelShape=labelShape)
		self.validationAugmenter = Transformer(["none"], dataShape=imageShape, labelShape=labelShape)
		self.setup()

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {Type : len(self.dataset[Type]["images"]) for Type in ("test", "train", "validation")}

		# These values are for cityscapes_v1
		# self.means = {
		# 	"images" : [72.0077217043444, 81.15868796015802, 70.88520810466197],
		# 	"depths" : 80.00175220926421,
		# 	"flownet2" : [-0.046412684, 2.4327936]
		# }

		# self.stds = {
		# 	"images" : [48.06690873921935, 48.75020861009473, 48.214420108835114],
		# 	"depths" : 64.32797789334958,
		# 	"flownet2" : [32.09688, 11.09397]
		# }

		# These values are for cityscapes_v2
		self.means = {
			"images" : [74.96715607296854, 84.3387139353354, 73.62945761147961],
			"depths" : 8277.619363028218,
			"flownet2" : [-0.6396361, 5.553444]
		}

		self.stds = {
			"images" : [49.65527668307159, 50.01892939272212, 49.67332749250472],
			"depths" : 6569.138224069467,
			"flownet2" : [32.508713, 15.168872]
		}

		print(("[CityScapes Images Reader] Setup complete. Num data: Train: %d, Test: %d, Validation: %d. " + \
			"Images shape: %s. Depths shape: %s. Flow algorithm: %s") % (self.numData["train"], self.numData["test"], \
			self.numData["validation"], self.imageShape, self.labelShape, self.flowAlgorithm))

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		thisData = self.dataset[type]

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = thisData["images"][startIndex : endIndex]
			depths = thisData["depths"][startIndex : endIndex]

			if not self.flowAlgorithm is None:
				flow = thisData[self.flowAlgorithm][startIndex : endIndex]
				images = np.concatenate([images, flow], axis=3)

			images, depths = self.normalizeData(np.float32(images), np.float32(depths))

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths

	def normalizeData(self, images, depths):
		images[..., 0] = standardizeData(images[..., 0], self.means["images"][0], self.stds["images"][0])
		images[..., 1] = standardizeData(images[..., 1], self.means["images"][1], self.stds["images"][1])
		images[..., 2] = standardizeData(images[..., 2], self.means["images"][2], self.stds["images"][2])
		depths = standardizeData(depths, self.means["depths"], self.stds["depths"])
		if not self.flowAlgorithm is None:
			images[..., 3] = standardizeData(images[..., 3], self.means[self.flowAlgorithm][0], \
				self.stds[self.flowAlgorithm][0])
			images[..., 4] = standardizeData(images[..., 4], self.means[self.flowAlgorithm][1], \
				self.stds[self.flowAlgorithm][1])
		return images, depths