import numpy as np
import os
import sys
from lycon import resize, Interpolation
from scipy.ndimage import gaussian_filter

def resize_batch(data, dataShape, type="bilinear"):
	# No need to do anything if shapes are identical.
	if data.shape[1 : ] == dataShape:
		return np.copy(data)

	assert type in ("bilinear", "nearest", "cubic")
	if type == "bilinear":
		interpolationType = Interpolation.LINEAR
	elif type == "nearest":
		interpolationType = Interpolation.NEAREST
	else:
		interpolationType = Interpolation.CUBIC

	numData = len(data)
	newData = np.zeros((numData, *dataShape), dtype=data.dtype)

	for i in range(len(data)):
		result = resize(data[i], height=dataShape[0], width=dataShape[1], interpolation=interpolationType)
		newData[i] = result.reshape(newData[i].shape)
	return newData

def resize_black_bars(data, desiredShape, type="bilinear"):
	# No need to do anything if shapes are identical.
	if data.shape == desiredShape:
		return np.copy(data)

	assert type in ("bilinear", "nearest", "cubic")
	if type == "bilinear":
		interpolationType = Interpolation.LINEAR
	elif type == "nearest":
		interpolationType = Interpolation.NEAREST
	else:
		interpolationType = Interpolation.CUBIC

	newData = np.zeros(desiredShape, dtype=data.dtype)
	# newImage = np.zeros((240, 320, 3), np.uint8)
	h, w = data.shape[0 : 2]
	desiredH, desiredW = desiredShape[0 : 2]

	# Find the rapports between the h/desiredH and w/desiredW
	rH, rW = h / desiredH, w / desiredW
	# print(rH, rW)

	# Find which one is the highest, that one will be used
	minRapp, maxRapp = min(rH, rW), max(rH, rW)
	if maxRapp == 0:
		return newData
	# print(minRapp, maxRapp)

	# Compute the new dimensions, based on th highest rapport
	newRh, newRw = int(h // maxRapp), int(w // maxRapp)
	# Also, find the half, so we can inser the other dimension from the half
	halfH, halfW = int((desiredH - newRh) // 2), int((desiredW - newRw) // 2)

	if newRw == 0 or newRh == 0:
		return newData

	resizedData = resize(data, height=newRh, width=newRw, interpolation=interpolationType)
	newData[halfH : halfH + newRh, halfW : halfW + newRw] = resizedData
	return newData

# Resizes a batch of HxW images, to a desired dHxdW, but keeps the same aspect ration, and adds black bars on the
#  dimension that does not fit (instead of streching as with regular resize).
def resize_batch_black_bars(data, desiredShape, type="bilinear"):
	# No need to do anything if shapes are identical.
	if data.shape[1 : ] == desiredShape:
		return np.copy(data)

	newData = np.zeros((numData, *desiredShape), dtype=data.dtype)
	for i in range(len(data)):
		newData[i] = resize_black_bars(data[i], desiredShape, type)

	return newData

def standardizeData(data, mean, std):
	data -= mean
	data /= std
	return data

def minMaxNormalizeData(data, min, max):
	data -= min
	data /= (max - min)
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

class LinePrinter:
	def __init__(self):
		self.maxLength = 0

	def print(self, message):
		if message[-1] == "\n":
			message = message[0 : -1]
			additional = "\n"
		else:
			additional = "\r"

		self.maxLength = np.maximum(len(message), self.maxLength)
		message += (self.maxLength - len(message)) * " " + additional
		sys.stdout.write(message)
		sys.stdout.flush()

# @brief Returns true if whatType is subclass of baseType. The parameters can be instantiated objects or types. In the
#  first case, the parameters are converted to their type and then the check is done.
def isBaseOf(whatType, baseType):
	if type(whatType) != type:
		whatType = type(whatType)
	if type(baseType) != type:
		baseType = type(baseType)
	return baseType in type(object).mro(whatType)

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
	return None if type(x) == type(None) else x if type(x) == list else [x]

# ["test"] and ["blah", "blah2"] => False
# ["blah2"] and ["blah", "blah2"] => True
def isSubsetOf(subset, set):
	for item in subset:
		if not item in set:
			return False
	return True

def changeDirectory(Dir, expectExist):
	assert os.path.exists(Dir) == expectExist
	print("Changing to working directory:", Dir)
	if expectExist == False:
		os.makedirs(Dir)
	os.chdir(Dir)

class RunningMean:
	def __init__(self):
		self.value = 0
		self.count = 0

	def update(self, value, count):
		if value != None:
			assert count > 0
			self.value += value
			self.count += count

	def get(self):
		return float(self.value / (self.count + 1e-5))

	def __repr__(self):
		return self.get()

	def __str__(self):
		return self.get()

# Given a graph as a dict {Node : [Dependencies]}, returns a list [Node] ordered with a correct topological sort order
# Applies Kahn's algorithm: https://ocw.cs.pub.ro/courses/pa/laboratoare/laborator-07
def topologicalSort(depGraph):
	print(depGraph)
	L, S = [], []

	# First step si to create a regular graph of {Node : [Children]}
	graph = {k : [] for k in depGraph.keys()}
	for key in depGraph:
		for parent in depGraph[key]:
			graph[parent].append(key)

	# Add nodes with no dependencies and start BFS from them
	depGraph = {k : len(depGraph[k]) for k in depGraph.keys()}
	for key in depGraph:
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
