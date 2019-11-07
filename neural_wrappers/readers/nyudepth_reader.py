import h5py
import numpy as np
from .dataset_reader import DatasetReader

# NYUDepthV2 only
class NYUDepthReader(DatasetReader):
	def __init__(self, datasetPath, imageShape, labelShape, transforms=["none"], \
		normalization="min_max_normalization", dataDimensions=["images"], labelDimensions=["depths"], \
		dataSplit=(80, 0, 20), semanticNumClasses=50, semanticTransform="none"):
		if "rgb" in dataDimensions:
			dataDimensions[dataDimensions.index("rgb")] = "images"

		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, labelDimensions, \
			transforms, normalization)
		self.dataSplit = dataSplit
		self.semanticNumClasses = semanticNumClasses
		self.semanticTransform = semanticTransform
		self.setup()

	def __str__(self):
		return "NYUDepth Reader"

	def semanticNewDims(self, images):
		numClasses = len(self.topClasses)
		newImages = np.zeros((numClasses, *images.shape), dtype=np.float32)
		for i in range(numClasses):
			thisOne = newImages[i]
			whereId = np.where(images == self.topClasses[i])
			thisOne[whereId] = 1
		newImages = np.transpose(newImages, [1, 3, 2, 0])
		return newImages

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		self.supportedDimensions = ["images", "depths", "labels"]
		numAllData = len(self.dataset["images"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)

		# Top 49 classes for each version of the dataset. The 50th is considered "other" and has value of 0.
		assert self.semanticNumClasses <= 49 and self.semanticNumClasses >= 1
		topClasses = np.array([21, 11, 3, 157, 5, 83, 28, 19, 59, 64, 88, 80, 7, 4, 36, 89, 42, 122, 169, 119, \
				143, 141, 85, 17, 172, 15, 123, 135, 26, 45, 331, 158, 144, 24, 124, 136, 55, 2, 238, 8, 1, 312, 94, \
				49, 242, 12, 87, 161, 298])
		self.topClasses = topClasses[0 : self.semanticNumClasses]

		self.postDataProcessing = {
			"images" : lambda x : np.swapaxes(x, 1, 3),
			"depths" : lambda x : np.swapaxes(x, 1, 2)
		}

		semanticNumDims = 1
		if "labels" in self.dataDimensions:
			assert self.semanticTransform in ("none", "semantic_new_dims")
			if self.semanticTransform == "none":
				prepareSemantic = lambda x : np.expand_dims(np.swapaxes(x, 1, 2), axis=-1)
			elif self.semanticTransform == "semantic_new_dims":
				semanticNumDims = self.semanticNumClasses
				prepareSemantic = self.semanticNewDims
			self.postDataProcessing["labels"] = prepareSemantic

		self.minimums = {
			"images" : [0, 0, 0],
			"depths" : 0,
			"labels" : [0] * semanticNumDims if semanticNumDims > 1 else 0
		}

		self.maximums = {
			"images" : [255, 255, 255],
			"depths" : 10,
			"labels" : [1] * semanticNumDims if semanticNumDims > 1 else 1
		}

		self.means = {
			"images" : [122.54539034418492, 104.78338963563233, 100.02394636162512],
			"depths" : 2.7963083,
			"labels" : [0] * semanticNumDims if semanticNumDims > 1 else 0
		}

		self.stds = {
			"images" : [73.74140829480243, 75.45561510079736, 78.87249644483357],
			"depths" : 1.3860533,
			"labels" : [1] * semanticNumDims if semanticNumDims > 1 else 1
		}

		self.numDimensions = {
			"images" : 3,
			"depths" : 1,
			"labels" : semanticNumDims
		}

		self.postSetup()
		print("[NYUDepth Reader] Setup complete. Num data: %d. Train: %d, Test: %d, Validation: %d" % \
			(numAllData, self.numData["train"], self.numData["test"], self.numData["validation"]))

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = self.indexes[type][0] + i * miniBatchSize
			endIndex = self.indexes[type][0] + min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			images = self.getData(self.dataset, startIndex, endIndex, self.dataDimensions)
			images = np.concatenate(images, axis=-1)
			depths = self.getData(self.dataset, startIndex, endIndex, self.labelDimensions)
			depths = np.concatenate(depths, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(images, depths, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths