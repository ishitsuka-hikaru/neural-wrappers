from functools import partial 

def makeGenerator(data, labels, batchSize:int, maxPrefetch:int=0):
	iteratorType = partial(NWIterator, data=data, labels=labels, batchSize=batchSize)
	return NWGenerator(iteratorType, maxPrefetch)

class NWIterator:
	def __init__(self, data, labels, batchSize:int):
		self.data = data
		self.labels = labels
		self.batchSize = batchSize
		self.ix = -1
	
	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			return self.__getitem__(self.ix)
		raise StopIteration

	def __getitem__(self, ix:int):
		startIndex = ix * self.batchSize
		endIndex = min(len(self.data), (ix + 1) * self.batchSize)
		if self.labels is None:
			X = (self.data[startIndex : endIndex], None)
		else:
			X = self.data[startIndex : endIndex], self.labels[startIndex : endIndex]
		return X, len(X)

	def __iter__(self):
		return self

	def __len__(self) -> int:
		N, B = len(self.data), self.batchSize
		n = N // B + (N % B != 0)
		return n

class NWGenerator:
	def __init__(self, iteratorType, maxPrefetch:int):
		assert maxPrefetch >= 0
		self.iteratorType = iteratorType
		self.maxPrefetch = maxPrefetch
		self.newEpoch()

	def newEpoch(self):
		self.currentIterator = self.iteratorType()
		self.currentLen = len(self.currentIterator)
		if self.maxPrefetch > 0:
			self.currentIterator = BackgroundGenerator(self.currentIterator, max_prefetch=self.maxPrefetch)
		# print("[iterateForever] New epoch. Len=%d. Batches: %s" % (self.currentLen, self.currentGenerator.batches))

	def __len__(self):
		return self.currentLen

	def __next__(self):
		try:
			return next(self.currentIterator)
		except StopIteration:
			self.newEpoch()
			return next(self.currentIterator)

	def __iter__(self):
		return self

	def __getitem__(self, key):
		return self.currentGenerator.__getitem__(key)

	def __getattr__(self, key):
		if isinstance(self.currentGenerator, BackgroundGenerator):
			return getattr(self.currentGenerator.generator, key)
		else:
			return getattr(self.currentGenerator, key)	