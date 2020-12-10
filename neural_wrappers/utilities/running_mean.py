import numpy as np
from typing import Union, Optional
from .utils import NWNumber, NWSequence, NWDict

class RunningMeanNumber:
	def __init__(self, initValue : NWNumber):
		self.value = initValue
		self.count = 0

	def update(self, value : NWNumber, count : Optional[int] = None):
		if not count:
			count = 1
		self.value += value
		self.count += count

	def updateBatch(self, value : NWSequence):
		value = np.array(value)
		assert len(value.shape) == 1
		self.update(value.sum(axis=0), value.shape[0])

	def get(self):
		assert self.count > 0
		return self.value / self.count

class RunningMeanSequence:
	def __init__(self, initValue : NWSequence):
		self.value = np.array(initValue)
		self.count = 0

	def update(self, value : NWSequence, count : Optional[int] = None):
		value = np.array(value)
		if not count:
			count = 1
		self.value += value
		self.count += count

	def updateBatch(self, value : NWSequence):
		value = np.array(value)
		assert len(value.shape) == len(self.value.shape) + 1
		self.update(value.sum(axis=0), value.shape[0])

	def get(self):
		assert self.count > 0
		return self.value / self.count

class RunningMeanDict:
	def __init__(self, initValue : NWDict):
		self.value = initValue
		self.count = 0

	def update(self, value : NWDict, count : Optional[int] = None):
		if not count:
			count = 1
		self.value = {k : self.value[k] + value[k] for k in self.value}
		self.count += count

	def updateBatch(self, value : NWDict):
		assert False, "Only valid for NWNumber and NWSequence"

	def get(self):
		assert self.count > 0
		return {k : self.value[k] / self.count for k in self.value}

class RunningMean:
	def __init__(self, initValue:Union[NWNumber, NWSequence, NWDict]):
		if type(initValue) in NWNumber.__args__: # type: ignore
			self.obj = RunningMeanNumber(initValue) # type: ignore
		elif type(initValue) in NWSequence.__args__: # type: ignore
			self.obj = RunningMeanSequence(initValue) # type: ignore
		elif type(initValue) in NWDict.__args__: # type: ignore
			self.obj = RunningMeanDict(initValue) # type: ignore
		else:
			print("[RunningMean] Doing a running mean on unknown type %s" % type(initValue))
			self.obj = RunningMeanNumber(initValue)

	def update(self, value:Union[NWNumber, NWSequence, NWDict], count:Optional[int] = 0):
		self.obj.update(value, count)

	def updateBatch(self, value : NWDict):
		self.obj.updateBatch(value)

	def get(self):
		return self.obj.get()

	def __repr__(self):
		return str(self.get())

	def __str__(self):
		return str(self.get())