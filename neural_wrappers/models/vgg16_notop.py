from torchvision.models import VGG
from torchvision.models.vgg import make_layers, cfg, model_urls
import torch.utils.model_zoo as model_zoo

# Class that loads a VGG-16 module from torchvision and deletes the FC layer at the end, to use just as extractor
class VGG16NoTop(VGG):
	def __init__(self, pretrained=True):
		super().__init__(make_layers(cfg['D']))
		if pretrained:
			self.load_state_dict(model_zoo.load_url(model_urls["vgg16"]))
		del self.classifier[-1]
