import h5py
import numpy as np
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class CitySimReader(DatasetReader):
	def __init__(self, dataGroup, datasetPath, imageShape, labelShape, transforms=["none"], \
		normalization="min_max_normalization", dataDimensions=["rgb"], labelDimensions=["depth"], **kwargs):
		super().__init__(datasetPath, imageShape, labelShape, dataDimensions, \
			labelDimensions, transforms, normalization)
		assert dataGroup in ("bragadiru_popesti", "london", "bucharest")
		self.dataGroup = dataGroup
		self.kwargs = kwargs
		self.allHvns = ["hvn_gt_raw", "hvn_gt_p1", "hvn_gt_p2", "hvn_gt_p3", "hvn_pred1_raw", "hvn_pred1_p1", \
			"hvn_pred1_p2", "hvn_pred1_p3", "hvn_pred2_raw"]
		self.setup()

	def __str__(self):
		return "CitySim Reader"

	def hvnTwoDims(self, images):
		hvn = np.zeros((2, *images.shape), dtype=np.float32)
		whereH = np.where(images == 0)
		whereV = np.where(images == 1)

		hvn[0][whereH] = 1
		hvn[1][whereV] = 1
		hvn = np.transpose(hvn, [1, 2, 3, 0])
		return hvn

	# This is a hack, the final solution involves in having a dictionary for normalizer as well, with default values
	#  defaulting to self.normalization, but with the possibility to update it (for example to self.doNothing) or
	#  any other more special normalizations.
	def minMaxNormalizer(self, data, type):
		if type in self.allHvns:
			return data
		else:
			return super().minMaxNormalizer(data, type)

	def standardizer(self, data, type):
		if type in self.allHvns:
			return data
		else:
			return super().standardizer(data, type)

	# Updates missing values w.r.t the dataGroup
	def setupMinMaxMeanStd(self):
		# TODO, maybe update dynamically when iterate_once is called based on type: train (bragadiru) / val (popesti)
		if self.dataGroup == "bragadiru_popesti":
			self.maximums["depth"] = 38.837765
			self.means["rgb"] = [121.64251106041375, 113.22833753162723, 110.21073242969062]
			self.means["depth"] = 11.086505
			self.stds["rgb"] = [55.31661791016009, 47.809744429727445, 45.23408344688476]
			self.stds["depth"] = 5.856089
		# London / ??
		elif self.dataGroup == "london":
			self.maximums["depth"] = 42.849
			self.means["rgb"] = [63.442951067503756, 63.460518690976265, 59.57283834466831]
			self.means["depth"] = 10.097478
			self.stds["rgb"] = [54.64836721734635, 51.51253766053043, 51.889279148779956]
			self.stds["depth"] = 5.6731715
		# Bucharest / ??
		elif self.dataGroup == "bucharest":
			self.maximums["depth"] = 39.4494
			self.means["rgb"] = [111.62589558919271, 112.83656475653362, 122.01291739800105]
			self.means["depth"] = 12.117755
			self.stds["rgb"] = [62.94051625776808, 55.35131957869077, 52.90895978272871]
			self.stds["depth"] = 5.330455
		else:
			assert False

		# HVN Setup - TODO update names
		hvnTransform = "none"
		for hvn in self.allHvns:
			if hvn in self.dataDimensions or hvn in self.labelDimensions:
				assert "hvnTransform" in self.kwargs
				hvnTransform = self.kwargs["hvnTransform"]
				break

		if hvnTransform == "none":
			hvnNumDims = 1
			hvnMax = 1
			hvnMin = 0
			hvnMean = 0
			hvnStd = 1
			hvnTransformFn = lambda x : np.expand_dims(x, axis=-1)
		elif hvnTransform == "hvn_two_dims":
			hvnNumDims = 2
			hvnMax = [1, 1]
			hvnMin = [0, 0]
			hvnMean = [0, 0]
			hvnStd = [1, 1]
			hvnTransformFn = self.hvnTwoDims
		else:
			assert False

		for hvn in self.allHvns:
			self.numDimensions[hvn] = hvnNumDims
			self.maximums[hvn] = hvnMax
			self.minimums[hvn] = hvnMin
			self.means[hvn] = hvnMean
			self.stds[hvn] = hvnStd
			self.postDataProcessing[hvn] = hvnTransformFn

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		# TODO, update p1, p2, p3 with actual names when they are finished
		# gt are ground truth values (from Blender)
		# pred1 are the outputs of the first network (rgb + gt => pred1)
		# pred2 are the outputs of the second network (rgb + pred1 => pred2)
		self.supportedDimensions = ("rgb", "depth", "hvn_gt_raw", "hvn_gt_p1", "hvn_gt_p2", "hvn_gt_p3", \
			"hvn_pred1_raw", "hvn_pred1_p1", "hvn_pred1_p2", "hvn_pred1_p3", "hvn_pred2_raw")

		# numData["train"] = N; numData["validation"] = M;
		self.numData = { "train": 0, "validation" : 0 }
		for Type in self.dataset.keys():
			self.numData[Type] = len(self.dataset[Type]["rgb"])

		self.minimums = { "rgb" : [0, 0, 0], "depth" : 0 }
		self.maximums = { "rgb" : [255, 255, 255] }
		self.means = { }
		self.stds = { }
		self.numDimensions = { "rgb" : 3, "depth" : 1 }
		self.setupMinMaxMeanStd()

		# Convert RGB from uint8 to float so we can normalize.
		self.postDataProcessing["rgb"] = lambda x : np.float32(x)
		self.postDataProcessing["depth"] = lambda x : np.expand_dims(x, axis=-1)

		self.postSetup()
		print("[CitySim Reader] Setup complete. Num data: (Train: %d, Validation: %d). Data dims: %s. Label dims: %s" \
			% (self.numData["train"], self.numData["validation"], self.dataDimensions, self.labelDimensions))

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter
		dataset = self.dataset[type]

		# One iteration in this method accounts for all transforms at once
		for i in range(self.getNumIterations(type, miniBatchSize, accountTransforms=False)):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			data = self.getData(dataset, startIndex, endIndex, self.dataDimensions)
			data = np.concatenate(data, axis=-1)
			labels = self.getData(dataset, startIndex, endIndex, self.labelDimensions)
			labels = np.concatenate(labels, axis=-1)

			# Apply each transform
			for augImages, augDepths in augmenter.applyTransforms(data, labels, interpolationType="bilinear"):
				yield augImages, augDepths
				del augImages, augDepths