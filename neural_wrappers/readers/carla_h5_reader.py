import numpy as np
import h5py
import os
import transforms3d.euler as txe
from functools import partial
from PIL import Image

from .dataset_reader import DatasetReader
from ..utilities import minMaxNormalizeData, tryReadImage

def getQuatFromRotation(rotation):
	assert rotation.min() >= -1 and rotation.max() <= 1
	# Move rotation from [-1 : 1] to [0 : 2*pi]
	roll, pitch, yaw = (rotation.T + 1) * np.pi
	quat = np.float32(np.concatenate([[txe.euler2quat(roll[i], pitch[i], yaw[i])] for i in range(len(roll))]))
	return quat

# Normalize inputs as desired
def poseNorm(positions, poseRepresentation, positionsExtremes, poseNormalization, inputSize):
	# Flatten the input, so we treat MBx6 and MBxseqSizex6 the same
	initialShape = positions.shape[0 : -1]
	positions = positions.reshape((-1, 6))

	# Positions are in coordinates (min/max) apply. Orientations are in [-180:180] range.
	assert poseRepresentation in ("6dof", "6dof-quat")
	minPos, maxPos = positionsExtremes["min"], positionsExtremes["max"]
	translation, rotation = positions[:, 0 : 3], positions[:, 3 :]

	# At this point translation should be between positionsExtremes and rotation in [-180:180]. We move them both in
	#  [-1 : 1] and then process the data further from this range.
	translation = (minMaxNormalizeData(translation, minPos, maxPos) - 0.5) * 2
	rotation = rotation / 180

	# [-1 : 1] representation all over the spectre (rotation + translation)
	if poseRepresentation == "6dof":
		result = np.concatenate([translation, rotation], axis=-1)
	# [-1 : 1] representation all over the spectre (rotation + 4 quaternion values)
	elif poseRepresentation == "6dof-quat":
		# quat::MBx4 which is already in [-1 : 1]
		quat = getQuatFromRotation(rotation)
		result = np.concatenate([translation, quat], axis=-1)
	else:
		assert False

	# Reshape to initial shape w.r.t first dimensions (MBxfinalDims or MBxseqSizexfinalDims)
	result = result.reshape((*initialShape, *result.shape[1 : ])).astype(np.float32)

	if poseNormalization == "min_max_-1_1":
		return result
	elif poseNormalization == "min_max_0_1":
		return result * 0.5 + 0.5

def depthNorm(x, depthNormalization, depthStats):
	x = np.clip(x, depthStats["min"], depthStats["max"])
	# print(depthStats)
	x = (x - depthStats["min"]) / (depthStats["max"] - depthStats["min"])
	if depthNormalization == "min_max_-1_1":
		return (x - 0.5) / 0.5
	elif depthNormalization == "min_max_0_1":
		return x

def depthRenorm(x, depthNormalization, depthStats):
	if depthNormalization == "min_max_-1_1":
		x = x * 0.5 + 0.5
	return x * (depthStats["max"] - depthStats["min"]) + depthStats["min"]

def rgbNorm(x, rgbNormalization):
	x = np.float32(x)
	if rgbNormalization == "min_max_-1_1":
		return (x / 255 - 0.5) / 0.5
	elif rgbNormalization == "min_max_0_1":
		return x / 255

def rgbRenorm(x, rgbNormalization):
	if rgbNormalization == "min_max_-1_1":
		x = (x * 0.5 + 0.5) * 255
	elif rgbNormalization == "min_max_0_1":
		x = x * 255
	return x.astype(np.int32)

def computeDistances(positions, numNeighbours):
	return # TODO
	if numNeighbours == 0:
		return None
	N = positions.shape[0]
	result = np.zeros((N, numNeighbours), dtype=np.int32) - 1
	for i in range(N):
		transDistances = np.linalg.norm(positions[i, 0 : 3] - positions[:, 0 : 3], axis=-1)
		rotDistances = np.linalg.norm(positions[i, 3 : ] - positions[:, 3 : ], axis=-1)
		# Put more weight on where they're watching rather than the actual position.
		allDistances = transDistances + 2 * rotDistances
		# Avoid first as it's always the distance to itself.
		allDistances[i] = np.inf
		argSortedDistances = np.argsort(allDistances)
		result[i] = argSortedDistances[0 : numNeighbours]
	# Enusre all values were filled.
	assert (result == -1).sum() == 0
	return result

def computeDepthSlices(depth, numSlices, depthNormalization):
	if depthNormalization == "min_max_-1_1":
		ranges = np.linspace(-1, 1, numSlices + 1)
	elif depthNormalization == "min_max_0_1":
		ranges = np.linspace(0, 1, numSlices + 1)
	lefts = ranges[0 : -1]
	rights = ranges[1 :]
	slices = np.zeros((numSlices, *depth.shape), dtype=np.float32)

	for i in range(numSlices - 1):
		left, right = lefts[i], rights[i]
		slices[i] = np.logical_and(depth >= left, depth < right)
	slices[-1] = (depth >= lefts[-1])
	slices = slices.transpose(1, 0, 2, 3)
	return slices

def computeIntrinsics(resolution, fieldOfView):
	fy, fx = resolution / (2 * np.tan(fieldOfView * np.pi / 360))
	cy, cx = np.array(resolution) / 2
	return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

class CarlaH5Reader(DatasetReader):
	def __init__(self, datasetPath, numNeighbours, poseRepresentation, fieldOfView=84, \
		numDepthSlices=0, dataDims = ["position"], labelDims=["rgb", "depth", "position"], \
		normalization = {"rgb" : "min_max_-1_1", "depth" : "min_max_-1_1", "position" : "min_max_-1_1"}, \
		resizer = {"rgb" : (320, 320, 3), "depth" : (320, 320)}):
		assert numNeighbours >= 0
		assert poseRepresentation in ("6dof", "6dof-quat", "dot_distance_transform", "dot")

		# Some stuff needs to be precomputed before call to parent, so we have lambdas ready.
		self.dataset = h5py.File(datasetPath, "r")
		self.poseRepresentation = poseRepresentation
		numData = {key : len(self.dataset[key]["rgb"]) for key in ["train", "validation"]}
		positions = {k : self.dataset[k]["position"][0 : numData[k]] for k in ["train", "validation"]}

		# Dataset specific data statistics (max depth, max XYZ and orientations etc.)
		stats = self.dataset["others"]["dataStatistics"]
		self.positionsExtremes = {k : stats["position"][k][0 : 3] for k in ["min", "max"]}
		self.rotationExtremes = {k : stats["position"][k][3 : ] for k in ["min", "max"]}
		self.depthStats = {k : stats["depth"][k][()] for k in stats["depth"]}
		self.resolution = resizer["rgb"][0 : 2]

		# Compute this here and not as a normalizer, because we need non-normalized values as labels.
		dimTransform = {
			"rgb" : partial(rgbNorm, rgbNormalization=normalization["rgb"]), \
			"depth" : partial(depthNorm, depthNormalization=normalization["depth"], depthStats=self.depthStats), \
			"position" : partial(poseNorm, poseRepresentation=self.poseRepresentation, inputSize=self.resolution, \
				positionsExtremes=self.positionsExtremes, poseNormalization=normalization["position"])
		}

		super().__init__(datasetPath, dataDims=dataDims, labelDims=labelDims, \
			resizer=resizer, dimTransform=dimTransform)

		self.numData = numData
		self.numNeighbours = numNeighbours
		self.numDepthSlices = numDepthSlices

		# Compute distances between all positions => array of NxN
		self.topKNeighbours = {k : computeDistances(positions[k], self.numNeighbours) for k in positions}
		self.intrinsic = computeIntrinsics(self.resolution, fieldOfView)

		if numDepthSlices > 1:
			self.dimTransform["depthSlices"] = partial(computeDepthSlices, \
				numSlices=self.numDepthSlices, depthNormalization=normalization["depth"])

	def iterate_once(self, type, miniBatchSize):
		dataset = self.dataset[type]
		numIterations = self.getNumIterations(type, miniBatchSize, accountTransforms=False)
		thisNeighbours = self.topKNeighbours[type]

		for i in range(numIterations):
			startIndex, endIndex = i * miniBatchSize, min((i + 1) * miniBatchSize, self.numData[type])
			assert startIndex < endIndex, "startIndex < endIndex. Got values: %d %d" % (startIndex, endIndex)
			data, labels = self.getData(dataset, startIndex, endIndex)

			if self.numNeighbours:
				neighboursIndexes = thisNeighbours[startIndex : endIndex]
				neighbourPositions = self.getDataFromH5(dataset, "position", neighboursIndexes)
				labels["neighbourPositions"] = neighbourPositions.copy()
				data["neighbourPositions"] = self.dataFinalTransform["position"](neighbourPositions)
				labels["neighbourRgbs"] = self.getDataFromH5(dataset, "rgb", neighboursIndexes)
				labels["neighbourDepths"] = self.getDataFromH5(dataset, "depth", neighboursIndexes)

			if self.numDepthSlices > 1:
				labels["depthSlices"] = self.dimTransform["depthSlices"](labels["depth"])

			yield data, labels

class CarlaH5PathsReader(CarlaH5Reader):
	@staticmethod
	def unrealFloatFronPng(x):
		x
		x = (x[..., 0] + x[..., 1] * 256 + x[..., 2] * 256 * 256) / (256 * 256 * 256 - 1)
		return x.astype(np.float32)

	@staticmethod
	def doPng(path, baseDirectory):
		path = baseDirectory + os.sep + str(path, "utf8")
		npImg = tryReadImage(path).astype(np.uint8)
		return npImg

	@staticmethod
	def doDepth(path, baseDirectory):
		path = baseDirectory + os.sep + str(path, "utf8")
		dph = tryReadImage(path)
		dph = CarlaH5PathsReader.unrealFloatFronPng(dph) * 1000
		return np.expand_dims(dph, axis=-1)

	@staticmethod
	def doOpticalFlow(path, baseDirectory):
		def readFlow(path):
			x = tryReadImage(path)
			# x :: [0 : 1]
			x = CarlaH5PathsReader.unrealFloatFronPng(x)
			# x :: [-1 : 1]
			x = (x - 0.5) * 2
			return x

		path_x, path_y = list(map(lambda x : "%s/%s" % (baseDirectory, str(x, "utf8")), path))
		flow_x, flow_y = readFlow(path_x), readFlow(path_y)
		flow = np.array([flow_x, flow_y]).transpose(1, 2, 0)
		return flow

	@staticmethod
	def doSemantic(path, baseDirectory):
		item = CarlaH5PathsReader.doPng(path, baseDirectory)
		labels = {
			(0, 0, 0): "Unlabeled",
			(70, 70, 70): "Building",
			(153, 153, 190): "Fence",
			(160, 170, 250): "Other",
			(60, 20, 220): "Pedestrian",
			(153, 153, 153): "Pole",
			(50, 234, 157): "Road line",
			(128, 64, 128): "Road",
			(232, 35, 244): "Sidewalk",
			(35, 142, 107): "Vegetation",
			(142, 0, 0): "Car",
			(156, 102, 102): "Wall",
			(0, 220, 220): "Traffic sign"
		}
		labelKeys = list(labels.keys())
		result = np.zeros(shape=item.shape[0] * item.shape[1], dtype=np.uint8)
		flattenedRGB = item.reshape(-1, 3)

		for i, label in enumerate(labelKeys):
			equalOnAllDims = np.prod(flattenedRGB == label, axis=-1)
			where = np.where(equalOnAllDims == 1)[0]
			result[where] = i

		result = result.reshape(*item.shape[0 : 2])
		return result

	# Normals are stored as [0 - 255] on 3 channels, representing the normals w.r.t world. We move them to [-1 : 1]
	@staticmethod
	def doNormal(path, baseDirectory):
		item = CarlaH5PathsReader.doPng(path, baseDirectory)
		item = ((np.float32(item) / 255) - 0.5) * 2
		return item

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		baseDirectory = self.dataset["others"]["baseDirectory"][()]
		self.dimGetter["rgb"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doPng(path, baseDirectory) for path in dataset["rgb"][startIndex : endIndex]])
		self.dimGetter["wireframe"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doPng(path, baseDirectory) \
			for path in dataset["wireframe"][startIndex : endIndex]])
		self.dimGetter["halftone"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doPng(path, baseDirectory) \
			for path in dataset["halftone"][startIndex : endIndex]])
		self.dimGetter["depth"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doDepth(path, baseDirectory) \
			for path in dataset["depth"][startIndex : endIndex]])
		self.dimGetter["semantic_segmentation"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doSemantic(path, baseDirectory) \
			for path in dataset["semantic_segmentation"][startIndex : endIndex]])
		self.dimGetter["normal"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doNormal(path, baseDirectory) \
			for path in dataset["normal"][startIndex : endIndex]])
		self.dimGetter["cameranormal"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doNormal(path, baseDirectory) \
			for path in dataset["cameranormal"][startIndex : endIndex]])
		self.dimGetter["rgbDomain2"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doPng(path, baseDirectory) \
			for path in dataset["rgbDomain2"][startIndex : endIndex]])
		self.dimGetter["optical_flow"] = lambda dataset, dim, startIndex, endIndex: \
			np.array([CarlaH5PathsReader.doOpticalFlow(path, baseDirectory) \
			for path in dataset["optical_flow"][startIndex : endIndex]])
