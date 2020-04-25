from neural_wrappers.readers import DatasetReader as Reader
import numpy as np

class TestDatasetReader:
	def test_constructor_1(self):
		reader = Reader(
			allDims = {
				"data" : ["rgb", "depth"],
				"labels" : ["depth", "semantic"]
			},
			dimGetter = {
				"rgb" : lambda a, b: None,
				"depth" : lambda a, b: None,
				"semantic" : lambda a, b: None
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : x,
					"depth" : lambda x : x
				},
				"labels" : {
					"depth" : lambda x : x,
					"semantic" : lambda x : x
				}
			}
		)

	def test_constructor_2(self):
		reader = Reader(
			allDims = {
				"data" : ["rgb", "depth"],
				"labels" : ["depth", "semantic"]
			},
			dimGetter = {
				"rgb" : lambda a, b: None,
				"depth" : lambda a, b: None,
				"semantic" : lambda a, b: None
			},
			dimTransform = {
				"data" : {
					"rgb" : lambda x : x,
				},
				"labels" : {
					"semantic" : lambda x : x
				}
			}
		)

def main():
	TestDatasetReader().test_constructor_1()
	TestDatasetReader().test_constructor_2()

if __name__ == "__main__":
	main()