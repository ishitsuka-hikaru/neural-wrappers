# from batched_dataset_reader.test_batched_dataset_reader import Dataset
# from neural_wrappers.readers import PercentDatasetReader
# from neural_wrappers.utilities import npCloseEnough

# class TestPercentDatasetReader:
# 	def test_constructor_1(self):
# 		for i in range(1, 100):
# 			reader = PercentDatasetReader(Dataset(), percent=i)

# 		try:
# 			reader = PercentDatasetReader(Dataset(), percent=0)
# 			assert False
# 		except AssertionError:
# 			pass

# 		try:
# 			reader = PercentDatasetReader(Dataset(), percent=101)
# 			assert False
# 		except AssertionError:
# 			pass


# 	def test_getNumData_1(self):
# 		for i in range(1, 100):
# 			reader = PercentDatasetReader(Dataset(), percent=i)
# 			assert reader.getNumData("train") == i

# 	def test_getNumIterations_1(self):
# 		for i in range(1, 100):
# 			reader = PercentDatasetReader(Dataset(), percent=i)
# 			assert reader.getNumIterations("train", 1) == i

# 	def test_getNumIterations_2(self):
# 		for i in range(1, 100):
# 			reader = PercentDatasetReader(Dataset(), percent=i)
# 			assert reader.getNumIterations("train", i) == 1

# 	def test_getNumIterations_3(self):
# 		for i in range(1, 100):
# 			reader = PercentDatasetReader(Dataset(), percent=i)
# 			assert reader.getNumIterations("train", 2) == i // 2 + (i % 2 == 1)

# 	def test_iterate_1(self):
# 		reader = PercentDatasetReader(Dataset(), percent=50)
# 		steps = reader.getNumIterations("train", 9)
# 		assert steps == 6
# 		generator = reader.iterate("train", 9)
# 		assert not generator is None
# 		items = next(generator)
# 		assert not items is None
# 		assert list(items.keys()) == ["data", "labels"]
# 		assert list(items["data"]) == ["D1", "D2"]
# 		assert list(items["labels"]) == ["D2", "D3"]

# 		assert npCloseEnough(items["data"]["D1"], reader.dataset["train"]["D1"][0 : 9])
# 		assert npCloseEnough(items["data"]["D2"], reader.dataset["train"]["D2"][0 : 9])
# 		assert npCloseEnough(items["labels"]["D3"], reader.dataset["train"]["D3"][0 : 9])
# 		assert items["labels"]["D2"].sum() == 0

# def main():
# 	TestPercentDatasetReader().test_constructor_1()

# if __name__ == "__main__":
# 	main()