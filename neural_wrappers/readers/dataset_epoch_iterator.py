from .dataset_reader import DatasetReader

# Iterator that iterates one epoch over this dataset.
# @brief Epoch iterator that goes through the provided dataset reader for exactly one epoch as defined by len(reader)
# @param[in] reader The DatasetReader we are iterating one epoch upon
class DatasetEpochIterator:
	def __init__(self, reader:DatasetReader):
		self.reader = reader
		self.ix = -1
		self.len = len(self.reader)
	
	def __len__(self):
		return self.len

	# @brief a function that maps a numeric index to the numeric index of the current epoch's item. By default
	#  f(x) = x, but some datasets may want to update this mapping for more sophisticated indexing algorithms
	def getIndexMapping(self, ix):
		return ix

	def __next__(self):
		self.ix += 1
		if self.ix < len(self):
			index = self.getIndexMapping(self.ix)
			return self.reader[index]
		raise StopIteration

	def __iter__(self):
		return self
