from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, model_urls
import torch.utils.model_zoo as model_zoo

# Class that loads a ResNet-50 module from torchvision and deletes the FC layer at the end, to use just as extractor
class ResNet50NoTop(ResNet):
	def __init__(self):
		super().__init__(Bottleneck, [3, 4, 6, 3])
		self.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
		self.fc = None
		self.avgpool = None

	# Overwrite the ResNet's forward phase to drop the average pooling and the fully connected layers
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x