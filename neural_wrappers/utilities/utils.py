import numpy as np
import os
import pickle
from collections import OrderedDict
from .np_utils import npCloseEnough
from .type_utils import Number, Dict, Sequence, Union, isBaseOf

def standardizeData(data, mean, std):
	data -= mean
	data /= std
	return data

def minMaxNormalizeData(data, min, max):
	data -= min
	data /= (max - min)
	data[data != data] = 0
	data[np.isinf(data)] = 0
	return data

def toCategorical(data, numClasses):
	numData = len(data)
	newData = np.zeros((numData, numClasses), dtype=np.uint8)
	newData[np.arange(numData), data] = 1
	return newData

# Labels can be None, in that case only data is available (testing cases without labels)
def makeGenerator(data, labels, batchSize):
	while True:
		numData = data.shape[0]
		numIterations = numData // batchSize + (numData % batchSize != 0)
		for i in range(numIterations):
			startIndex = i * batchSize
			endIndex = np.minimum((i + 1) * batchSize, numData)
			if not labels is None:
				yield data[startIndex : endIndex], labels[startIndex : endIndex]
			else:
				yield data[startIndex : endIndex]

def NoneAssert(conndition, noneCheck, message=""):
	if noneCheck:
		assert conndition, message

# Stubs for identity functions, first is used for 1 parameter f(x) = x, second is used for more than one parameter,
#  such as f(x, y, z) = (x, y, z)
def identity(x, **kwargs):
	return x

def identityVar(*args):
	return args

# Stub for making a list, used by various code parts, where the user may provide a single element for a use-case where
#  he'd have to use a 1-element list. This handles that case, so the overall API uses lists, but user provides
#  just an element. If None, just return None.
def makeList(x):
	return None if type(x) == type(None) else list(x) if type(x) in (list, set, tuple) else [x]

# ["test"] and ["blah", "blah2"] => False
# ["blah2"] and ["blah", "blah2"] => True
def isSubsetOf(subset, set):
	for item in subset:
		if not item in set:
			return False
	return True

def changeDirectory(Dir, expectExist=None):
	if expectExist in (True, False):
		assert os.path.exists(Dir) == expectExist
	print("Changing to working directory:", Dir)
	if expectExist == False or (expectExist == None and not os.path.isdir(Dir)):
		os.makedirs(Dir)
	os.chdir(Dir)

# Given a graph as a dict {Node : [Dependencies]}, returns a list [Node] ordered with a correct topological sort order
# Applies Kahn's algorithm: https://ocw.cs.pub.ro/courses/pa/laboratoare/laborator-07
def topologicalSort(depGraph):
	L, S = [], []

	# First step is to create a regular graph of {Node : [Children]}
	graph = {k : [] for k in depGraph.keys()}
	for key in depGraph:
		for parent in depGraph[key]:
			graph[parent].append(key)
		# Transform the depGraph into a list of number of in-nodes
		depGraph[key] = len(depGraph[key])
		# Add nodes with no dependencies and start BFS from them
		if depGraph[key] == 0:
			S.append(key)

	while len(S) > 0:
		u = S.pop()
		L.append(u)

		for v in graph[u]:
			depGraph[v] -= 1
			if depGraph[v] == 0:
				S.append(v)

	for key in depGraph:
		if depGraph[key] != 0:
			raise Exception("Graph is not acyclical")
	return L

def getGenerators(reader, miniBatchSize, maxPrefetch=1, keys=["train", "validation"]):
	items = []
	for key in keys:
		generator = reader.iterate(key, miniBatchSize=miniBatchSize, maxPrefetch=maxPrefetch)
		numIters = reader.getNumIterations(key, miniBatchSize=miniBatchSize)
		items.extend([generator, numIters])
	return items

def tryReadImage(path, count=5, imgLib="opencv"):
	assert imgLib in ("opencv", "PIL", "lycon")

	def readImageOpenCV(path):
		import cv2
		bgr_image = cv2.imread(path)
		b, g, r = cv2.split(bgr_image)
		image = cv2.merge([r, g, b]).astype(np.float32)
		return image

	def readImagePIL(path):
		from PIL import Image
		image = np.array(Image.open(path), dtype=np.float32)[..., 0 : 3]
		return image

	def readImageLycon(path):
		from lycon import load
		image = load(path)[..., 0 : 3].astype(np.float32)
		return image

	f = {
		"opencv" : readImageOpenCV,
		"PIL" : readImagePIL,
		"lycon" : readImageLycon
	}[imgLib]

	i = 0
	while True:
		try:
			return f(path)
		except Exception as e:
			print("Path: %s. Exception: %s" % (path, e))
			i += 1

			if i == count:
				raise Exception

# Deep check if two items are equal. Dicts are checked value by value and numpy array are compared using "closeEnough"
#  method
def deepCheckEqual(a, b):
	if type(a) != type(b):
		print("Types %s and %s differ." % (type(a), type(b)))
		return False
	Type = type(a)
	if Type in (dict, OrderedDict):
		for key in a:
			if not deepCheckEqual(a[key], b[key]):
				return False
		return True
	elif Type == np.ndarray:
		return npCloseEnough(a, b)
	else:
		return a == b
	assert False, "Shouldn't reach here"
	return False

def isPicklable(item):
	try:
		_ = pickle.dumps(item)
		return True
	except Exception as e:
		print("Item is not pickable: %s" % (e))
		return False

# Flatten the indexes [[1, 3], [15, 13]] => [1, 3, 15, 13] and then calls f(data, 1), f(data, 3), ..., step by step
def smartIndexWrapper(data, indexes, f):
	# Flatten the indexes [[1, 3], [15, 13]] => [1, 3, 15, 13]
	indexes = np.array(indexes, dtype=np.uint32)
	flattenedIndexes = indexes.flatten()
	N = len(flattenedIndexes)
	assert N > 0

	# Get the first element to infer shape
	firstResult = f(data, flattenedIndexes[0])
	flattenedShape = (N, *firstResult.shape)
	finalShape = (*indexes.shape, *firstResult.shape)

	result = np.zeros(flattenedShape, firstResult.dtype)
	result[0] = firstResult
	for i in range(1, N):
		result[i] = f(data, flattenedIndexes[i])
	return result.reshape(finalShape)

def getFormattedStr(item : Union[np.ndarray, Number, Sequence, Dict], precision : int) -> str:
	formatStr = "%%2.%df" % (precision)
	if type(item) in Number.__args__:
		return formatStr % (item)
	elif type(item) in Sequence.__args__:
		return [formatStr % (x) for x in item]
	elif type(item) in Dict.__args__:
		return {k : formatStr % (item[k]) for k in item}
	assert False, "Unknown type: %s" % (type(item))

def flattenList(x):
	return sum(x, [])
