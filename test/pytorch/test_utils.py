from neural_wrappers.pytorch import maybeCuda
import torch as tr

class TestUtils:
	def test_maybeCuda_1(self):
		x = maybeCuda(tr.FloatTensor(10, 10))
		deviceStr = str(x.device)
		expectedStr = "cuda" if tr.cuda.is_available() else "cpu"
		assert expectedStr in deviceStr
