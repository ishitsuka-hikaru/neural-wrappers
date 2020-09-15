import torch as tr
import torch.nn as nn
from ..pytorch import FeedForwardNetwork

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

class ModelWord2Vec(FeedForwardNetwork):
	def __init__(self, dictionary, embeddingSize):
		super().__init__()
		self.dictionary = dictionary
		self.vocabSize = len(dictionary)
		self.embIn = nn.Embedding(num_embeddings=self.vocabSize, embedding_dim=embeddingSize)
		self.embOut = nn.Embedding(num_embeddings=self.vocabSize, embedding_dim=embeddingSize)

		# Set criterion to negative log likelihood cost function.
		self.setCriterion(lambda y, t : tr.mean(-tr.log(y[t] + 1e-5)))

	def networkAlgorithm(self, trInputs, trLabels):
		posContext, negContexts = trLabels
		MB, N = trInputs.shape[0], negContexts.shape[1]

		wordEmb = self.embIn(trInputs)
		posEmb = self.embOut(posContext)
		negEmb = self.embOut(negContexts)

		# (MB, E) x (MB, E) => (MB, 1, E) x (MB, E, 1) => (MB, 1, 1) => (MB, 1)
		pos = tr.bmm(wordEmb.unsqueeze(1), posEmb.unsqueeze(-1)).squeeze(-1)
		# (MB, N, E) x (MB, E) => (MB, N, E) x (MB, E, 1) => (MB, N, 1) => (MB, N)
		neg = tr.bmm(negEmb, wordEmb.unsqueeze(-1)).squeeze(-1)

		# Positive word must be a big number (so sigmoid truncates it to 1), while negative context wods must be
		#  small numbers, so sigmoid truncates it to 0. If we negate the negative number, all of them will be large in
		#  the correct case. The labels are just a bunch of ones, which are then put through a NLL cost function.
		both = tr.sigmoid(tr.cat([pos, -neg], dim=-1))
		labels = tr.ones(MB, N + 1).byte().to(device)

		trLoss = self.criterion(both, labels)
		return both, trLoss

	def saveEmbeddings(self, outFile, quiet=False):
		vectors = self.embIn.weight.cpu().detach().numpy()
		f = open(outFile, "w")
		for i, word in enumerate(self.dictionary):
			if i % 100 == 0 and quiet == False:
				print("%d/%d words stored" % (i + 1, self.vocabSize))
			Str = word
			for x in vectors[i]:
				Str += " %2.5f" % (x)
			f.write(Str + "\n")
			f.flush()