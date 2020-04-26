from ...utilities import Sequence

# @biref Internal class used for indexing generic datasets dimensions
class DatasetIndex: pass

# @brief Internal class used for indexing ranges of type [start : end]
class DatasetRange(DatasetIndex):
	def __init__(self, start : int, end : int):
		self.start = start
		self.end = end

# @brief Internal class used for indexing with random "iterators" that preserve shapes:
#  [[1, 10], [2], [1, 2, [3, 5]]] shall return the values at those indices for this shape
class DatasetRandomIndex(DatasetIndex):
	def __init__(self, sequence : Sequence):
		self.sequence = sequence