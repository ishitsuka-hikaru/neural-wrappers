import numpy as np
from neural_wrappers.utilities import toCategorical

class Reader:
	def __init__(self, corpus, windowSize):
		self.corpus = corpus
		self.tokenized_corpus = self.tokenize_corpus()
		self.dictionary = self.getUniqueWords()
		self.vocabSize = len(self.dictionary)
		self.stoi = dict(zip(self.dictionary, range(len(self.dictionary))))
		self.windowSize = windowSize

	def tokenize_corpus(self):
		return [x.split() for x in self.corpus]

	def getUniqueWords(self):
		words = set()
		for sentence in self.tokenized_corpus:
			for word in sentence:
				words.add(word)
		return list(words)

	def getIndex(self, word):
		return self.stoi[word]

	def getWord(self, index):
		return self.dictionary[index]

	def getNumIterations(self, miniBatchSize=1):
		N = 0
		for sentence in self.tokenized_corpus:
			S, W = len(sentence), self.windowSize
			# Only computing left items context
			# Number of items that do not have full window
			# Example: "He is a king". S = 4, W = 2. "He" has 0 items to left
			#  "is" has 1 item to the left. => a = 0 + 1 = 1
			a = sum(range(min(W, S)))
			# Number of items with good left only context
			#  "a" has 2 items, "king" has 2 items => b = 4
			b = W * (S - W) * (S > W)
			# Total amount of context items is doubled, because right items are symmetrical
			N += 2 * (a + b)
		return N // miniBatchSize + (N % miniBatchSize != 0)

	def iterate_words(self, miniBatchSize):
		words, contexts = [], []
		N = self.getNumIterations(miniBatchSize)
		step = 0
		for sentence in self.tokenized_corpus:
			indices = [self.getIndex(w) for w in sentence]
			for i in range(len(sentence)):
				center_word = indices[i]
				for w in range(-self.windowSize, self.windowSize + 1):
					step += 1
					j = i + w
					if j < 0 or j >= len(sentence) or j == i:
						continue
					context_word = indices[j]

					words.append(center_word)
					contexts.append(context_word)
					if len(words) == miniBatchSize or step == N:
						yield np.array(words), np.array(contexts)
						words.clear()
						contexts.clear()

	def iterate_once(self, miniBatchSize):
		for item in self.iterate_words(miniBatchSize):
			words, contexts = item
			contexts = toCategorical(np.array(contexts), numClasses=self.vocabSize)
			yield words, contexts

	def iterate(self, miniBatchSize):
		while True:
			iterateGenerator = self.iterate_once(miniBatchSize)
			for items in iterateGenerator:
				words, contexts = items
				yield items
				del items

class Word2VecReader(Reader):
	def __init__(self, corpus, windowSize, numNegative):
		super().__init__(corpus, windowSize)
		self.numNegative = numNegative

	def iterate_once(self, miniBatchSize):
		generator = super().iterate_words(miniBatchSize)
		for words, contexts in generator:
			negativeContexts = np.random.randint(0, self.vocabSize, size=(len(words), self.numNegative))
			yield words, (contexts, negativeContexts)