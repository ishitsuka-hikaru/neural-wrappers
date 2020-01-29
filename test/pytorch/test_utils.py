from neural_wrappers.pytorch.utils import device
import torch as tr

class TestUtils:
	def test_device_1(self):
		x = tr.FloatTensor(10, 10).to(device)
		deviceStr = str(x.device)
		expectedStr = "cuda" if tr.cuda.is_available() else "cpu"
		assert expectedStr in deviceStr
