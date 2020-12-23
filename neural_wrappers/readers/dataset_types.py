import numpy as np
from typing import Union, Callable, Sequence, Any

# @brief Internal class used for indexing with random "iterators" that preserve shapes:
#  [[1, 10], [2], [1, 2, [3, 5]]] shall return the values at those indices for this shape
class DatasetRandomIndex:
	def __init__(self, sequence:Sequence):
		self.sequence = sequence

	def __str__(self):
		return "DatasetRandomIndex: %s" % (str(self.sequence))

DatasetIndex = Union[int, Sequence[int], np.ndarray, range, DatasetRandomIndex]
#DimGetterParams = [str, DatasetIndex]
DimGetterCallable = Callable[[str, DatasetIndex], np.ndarray]
#DimTransformParams = [str, np.ndarray]
DimTransformCallable = Callable[[str, np.ndarray], np.ndarray]
DatasetItem = Any