from neural_wrappers.readers import DatasetReaderV2 as Reader
import numpy as np

class TestDatasetReaderV2:
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