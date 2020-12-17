import numpy as np
from overrides import overrides
from typing import Any, Iterator, Dict, Callable
from neural_wrappers.readers import CompoundDatasetReader, DatasetIndex
from test_dataset_reader import DummyDataset

class TestCompoundDatasetReader:
	def test_constructor_1(self):
		reader = CompoundDatasetReader(DummyDataset())
		assert not reader is None
		assert reader.datasetFormat.dataBuckets == {"data" : ["rgb"], "labels" : ["class"]}
		assert list(reader.datasetFormat.dimGetter.keys()) == ["rgb", "class"]
		for v in reader.datasetFormat.dimGetter.values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform.keys()) == ["data", "labels"]
		assert list(reader.datasetFormat.dimTransform["data"].keys()) == ["rgb"]
		for v in reader.datasetFormat.dimTransform["data"].values():
			assert isinstance(v, Callable)
		assert list(reader.datasetFormat.dimTransform["labels"].keys()) == ["class"]
		for v in reader.datasetFormat.dimTransform["labels"].values():
			assert isinstance(v, Callable)

def main():
	TestCompoundDatasetReader().test_constructor_1()

if __name__ == "__main__":
	main()