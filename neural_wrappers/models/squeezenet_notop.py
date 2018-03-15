from torchvision.models import SqueezeNet
from torchvision.models.squeezenet import model_urls
import torch.utils.model_zoo as model_zoo

# Class that loads a ResNet-50 module from torchvision and deletes the FC layer at the end, to use just as extractor
class SqueezeNetNoTop(SqueezeNet):
	def __init__(self):
		super().__init__(version=1.1)
		self.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
		self.classifier = None

	# Overwrite the ResNet's forward phase to drop the average pooling and the fully connected layers
	def forward(self, x):
		return self.features(x)