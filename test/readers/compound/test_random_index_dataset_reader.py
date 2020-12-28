import numpy as np
import pycache
from overrides import overrides
from typing import Tuple, List, Any
from neural_wrappers.readers import CachedDatasetReader, DatasetItem, DatasetIndex, RandomIndexDatasetReader
from neural_wrappers.utilities import getGenerators, deepCheckEqual

import sys
sys.path.append("..")
from batched_dataset_reader.test_batched_dataset_reader import Reader as BaseBatchedReader
from test_dataset_reader import DummyDataset as BaseReader

class TestRandomIndexDatasetReader:
	def test_constructor_1(self):
		reader = RandomIndexDatasetReader(BaseReader(), seed=42)
		assert not reader is None

	def test_iterateForever_1(self):
		reader = RandomIndexDatasetReader(BaseReader(), seed=42)
		generator = reader.iterateForever()
		N = 10
		items = []
		for i in range(N):
			epochItems = []
			for j in range(len(generator)):
				epochItems.append(next(generator))
			items.append(epochItems)

		# len(reader)^10 == 10^10 chance of this test not passing :)
		firstItems = items[0]
		equal = []
		for i in range(1, N):
			ixItems = items[i]
			equal.append(deepCheckEqual(firstItems, ixItems))
		assert sum(equal) != N - 1

class TestRandomIndexBatchedDatasetReader:
	def test_constructor_1(self):
		reader = RandomIndexDatasetReader(BaseBatchedReader(), seed=42)
		assert not reader is None

	def test_iterateForever_1(self):
		reader = RandomIndexDatasetReader(BaseBatchedReader(), seed=42)
		generator = reader.iterateForever()
		N = 10
		items = []
		for i in range(N):
			epochItems = []
			for j in range(len(generator)):
				epochItems.append(next(generator))
			items.append(epochItems)
		# len(reader)^10 == 10^10 chance of this test not passing :)
		firstItems = items[0]
		equal = []
		for i in range(1, N):
			ixItems = items[i]
			equal.append(deepCheckEqual(firstItems, ixItems))
		assert sum(equal) != N - 1

def main():
	TestRandomIndexBatchedDatasetReader().test_iterateForever_1()

if __name__ == "__main__":
	main()