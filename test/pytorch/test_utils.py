from neural_wrappers.pytorch import maybeCuda
import torch as tr

class TestUtils:
	def test_maybeCuda_1(self):
		x = tr.FloatTensor(10, 10)
		x_cuda = maybeCuda(x)
		if tr.cuda.is_available():
			assert type(x_cuda) == tr.cuda.FloatTensor
		else:
			assert type(x_cuda) == tr.FloatTensor