import os  # Importing the os module to interact with the operating system
import torch  # Importing PyTorch, a popular machine learning library

# Dictionary class to map words to indices and vice versa
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}  # A dictionary to store word to index mappings
        self.idx2word = []  # A list to store index to word mappings
        self.count = {}     # A dictionary to count occurrences of each word

    # Function to add a word to the dictionary
    def add_word(self, word):
        word = word.lower()  # Convert word to lowercase for consistency
        if word not in self.word2idx:  # If word is not in dictionary
            self.idx2word.append(word)  # Add word to the index list
            self.word2idx[word] = len(self.idx2word) - 1  # Map word to index
            self.count[word] = 1  # Initialize word count to 1
        else:
            self.count[word] = self.count[word] + 1  # Increment word count if already exists
        return self.word2idx[word]  # Return the index of the word

    # Function to get the index of a word
    def index_word(self, word):
        word = word.lower()  # Convert word to lowercase
        if word in self.word2idx:
            return self.word2idx[word]  # Return index if word is in the dictionary
        else:
            return self.__len__() - 1  # Return the length of the dictionary if word not found

    # Function to get the size of the dictionary
    def __len__(self):
        return len(self.idx2word) + 1  # Length of the index list plus one

# Corpus class to handle train, validation, and test datasets
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()  # Initialize a Dictionary object
        self.train = self.tokenize(os.path.join(path, 'train.txt'))  # Tokenize training data
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))  # Tokenize validation data
        self.test = self.tokenize(os.path.join(path, 'test.txt'))    # Tokenize test data

    # Function to tokenize a text file
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)  # Ensure the file exists
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']  # Split line into words and add end-of-sequence token
                tokens += len(words)  # Count the number of tokens
                for word in words:
                    self.dictionary.add_word(word)  # Add each word to the dictionary

        # Tokenize file content into indices
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)  # Create a tensor to hold the word indices
            token = 0
            for line in f:
                words = line.split() + ['<eos>']  # Split line into words and add end-of-sequence token
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]  # Convert word to index
                    token += 1  # Move to the next token

        return ids  # Return the tensor of tokenized words
