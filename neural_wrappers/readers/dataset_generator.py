from prefetch_generator import BackgroundGenerator
from .dataset_reader import DatasetReader

# @brief Infinite generator that goes, epoch by epoch, through the provided dataset reader.
# @param[in] reader The DatasetReader we are iterating forever upon
# @param[in] maxPrefetch The number of threads (based on BackgroundGenerator module) to use for iterating purposes
# TODO: See if can be replaced by utilities/NWGenerator
class DatasetGenerator:
	def __init__(self, reader:DatasetReader, maxPrefetch:int):
		assert maxPrefetch >= 0
		self.reader = reader
		self.maxPrefetch = maxPrefetch
		self.newEpoch()

	def newEpoch(self):
		self.currentGenerator = self.reader.iterateOneEpoch()
		self.currentLen = len(self.currentGenerator)
		if self.maxPrefetch > 0:
			self.currentGenerator = BackgroundGenerator(self.currentGenerator, max_prefetch=self.maxPrefetch)
		# print("[iterateForever] New epoch. Len=%d. Batches: %s" % (self.currentLen, self.currentGenerator.batches))

	def __len__(self):
		return self.currentLen

	def __next__(self):
		try:
			return next(self.currentGenerator)
		except StopIteration:
			self.newEpoch()
			return next(self.currentGenerator)

	def __iter__(self):
		return self

	def __getitem__(self, key):
		return self.currentGenerator.__getitem__(key)

	def __getattr__(self, key):
		if isinstance(self.currentGenerator, BackgroundGenerator):
			return getattr(self.currentGenerator.generator, key)
		else:
			return getattr(self.currentGenerator, key)