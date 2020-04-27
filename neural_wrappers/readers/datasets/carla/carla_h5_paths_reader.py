from ....h5_dataset_reader import H5DatasetReader

class CarlaH5PathsReader(H5DatasetReader):
	def __init__(self, datasetPath : str, dataBuckets : Dict[str, List[str]], \
		dimTransform : Dict[str, Dict[str, Callable]]):
		# dimGetter : Dict[str, DimGetterCallable], 

		dimGetter = {
			# "png"
		}

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
