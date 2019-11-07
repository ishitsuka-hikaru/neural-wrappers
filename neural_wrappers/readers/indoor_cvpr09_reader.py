import h5py
import numpy as np
from .dataset_reader import ClassificationDatasetReader

class IndoorCVPR09Reader(ClassificationDatasetReader):
	def __init__(self, datasetPath, imageShape, transforms=["none"], \
		normalization="min_max_normalization", dataSplit=(80, 0, 20)):
		super().__init__(datasetPath, imageShape, None, None, None, transforms, normalization)
		self.dataSplit = dataSplit
		self.setup()

	def __str__(self):
		return "Indoor-CVPR09 Reader"

	def getNumberOfClasses(self):
		return 67

	def getClasses(self):
		return ["airport_inside", "artstudio", "auditorium", "bakery", "bar", "bathroom", "bedroom", "bookstore", \
			"bowling", "buffet", "casino", "children_room", "church_inside", "classroom", "cloister", "closet", \
			"clothingstore", "computerroom", "concert_hall", "corridor", "deli", "dentaloffice", "dining_room", \
			"elevator", "fastfood_restaurant", "florist", "gameroom", "garage", "greenhouse", "grocerystore", "gym", \
			"hairsalon", "hospitalroom", "inside_bus", "inside_subway", "jewelleryshop", "kindergarden", "kitchen", \
			"laboratorywet", "laundromat", "library", "livingroom", "lobby", "locker_room", "mall", "meeting_room", \
			"movietheater", "museum", "nursery", "office", "operating_room", "pantry", "poolinside", "prisoncell", \
			"restaurant", "restaurant_kitchen", "shoeshop", "stairscase", "studiomusic", "subway", "toystore", \
			"trainstation", "tv_studio", "videostore", "waitingroom", "warehouse", "winecellar"]

	def setup(self):
		self.dataset = h5py.File(self.datasetPath, "r")
		numAllData = len(self.dataset["images"])
		self.indexes, self.numData = self.computeIndexesSplit(numAllData)
		self.supportedDimensions = ["images", "labels"]

		self.dataDimensions = ["images"]
		self.labelDimensions = ["labels"] # TODO

		self.numDimensions = {
			"images" : 3,
			"labels" : 1
		}

		self.means = {
			"images" : [103.44083498697917, 90.78795360677083, 77.51603048177083]
		}

		self.stds = {
			"images" : [77.17012269672304, 72.85816272357351, 69.99937519942654]
		}

		self.minimums = {
			"images" : [0, 0, 0]
		}

		self.maximums = {
			"images" : [255, 255, 255]
		}

		self.postSetup()
		print("[Indoor-CVPR09 Reader] Setup complete")

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
			labels = np.int64(self.dataset["labels"][startIndex : endIndex])

			# Apply each transform
			for augImages, _ in augmenter.applyTransforms(images, None, interpolationType="bilinear"):
				yield augImages, labels
