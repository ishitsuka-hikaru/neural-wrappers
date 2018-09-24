import numpy as np
import h5py
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

# # CityScapes Reader class, used with the data already converted in h5py format.
class CityScapesReader(DatasetReader):
	def __init__(self, datasetPath, dataDims, resizer, normalization="standardization"):
		super().__init__(datasetPath, \
			allDims = ["rgb", "depth"], \
			dataDims=dataDims, labelDims=["depth"], \
			dimTransform = {
				"rgb" : lambda x : np.float32(x),
				"depth" : lambda x : np.float32(x),
			}, \
			normalizer = {
				"rgb" : normalization,
				"depth" : normalization,
			}, \
			resizer=resizer)

		self.baseDataGroup = "noseq"
		self.dataset = h5py.File(self.datasetPath, "r")
		self.numData = {item : len(self.dataset[item][self.baseDataGroup]["rgb"]) for item in self.dataset}
		self.trainTransformer = self.transformer
		self.valTransformer = Transformer(self.allDims, [])

		self.maximums = {
			"rgb" : np.array([255, 255, 255]),
			"depth" : np.array([32257])
		}

		self.minimums = {
			"rgb" : np.array([0, 0, 0]),
			"depth" : np.array([0])
		}

		self.means = {
			"rgb" : np.array([74.96715607296854, 84.3387139353354, 73.62945761147961]),
			"depth" : np.array([8277.619363028218])
		}

		self.stds = {
			"rgb" : np.array([49.65527668307159, 50.01892939272212, 49.67332749250472]),
			"depth" : np.array([6569.138224069467]),
		}


	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "validation")
		if type == "train":
			self.transformer = self.trainTransformer
		else:
			self.transformer = self.valTransformer

		dataset = self.dataset[type][self.baseDataGroup]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)

		for i in range(numIterations):
			startIndex = i * miniBatchSize
			endIndex = min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)

			for items in self.getData(dataset, startIndex, endIndex):
				yield items

# Used for plotting, so we convert a [0-33] input image to a nice RGB semantic plot.
def cityscapesSemanticCmap(label):
	if label.ndim != 2:
		raise ValueError("Expect 2-D input label")

	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
	ind >>= 3

	if np.max(label) >= len(colormap):
		raise ValueError("label value too large.")

	return np.uint8(colormap[label])

