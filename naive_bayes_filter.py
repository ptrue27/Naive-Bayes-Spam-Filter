# Import libraries
import numpy as np
import pandas as pd
import argparse
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.training_set = None
        self.spam_set = None
        self.ham_set = None
        self.test_set = None
        self.n_spam = None
        self.p_spam = None
        self.n_ham = None
        self.p_ham = None
        self.n_vocabulary = None
        self.test_set_path = test_set_path
        self.stop_words = set(stopwords.words('english'))
        self.word_frequencies_spam = None
        self.word_frequencies_spam = None

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')

    def clean_text(self, text):
        # Remove URLs, email addresses, digits, and special characters
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\S+@\S+(?:\.\S+)+", "", text)
        text = re.sub(r"[^A-Za-z\s]", "", text)

        # Lower case, stem and remove stop words
        text = text.lower()
        ps = PorterStemmer()
        text = ' '.join([ps.stem(word) for word in set(text.split()) if word not in self.stop_words])

        return text

    def data_cleaning(self):
        # Normalization
        self.training_set['v2'] = self.training_set['v2'].apply(lambda x: self.clean_text(x))

        # Separate spam and ham data sets
        self.spam_set = self.training_set[self.training_set['v1'] == 'spam']
        self.ham_set = self.training_set[self.training_set['v1'] == 'ham']

    def get_word_frequencies(self, cls):
        # Get training data for specified class
        class_set = self.spam_set if cls =='spam' else self.ham_set

        # Find frequency of each word
        word_frequencies = {}
        for message in class_set['v2']:
            for word in message.split():
                if word in word_frequencies:
                    word_frequencies[word] = word_frequencies[word] + 1
                else:
                    word_frequencies[word] = 1

        return word_frequencies

    def fit_bayes(self):
        # Calculate P(Spam) and P(Ham)
        n = len(self.training_set)
        self.n_spam = len(self.spam_set)
        self.p_spam =  self.n_spam / n
        self.n_ham = len(self.ham_set)
        self.p_ham =  self.n_ham / n

        # Calculate Frequency(wi|Spam) and Frequency(wi|Ham)
        self.word_frequencies_spam = self.get_word_frequencies('spam')
        self.word_frequencies_ham = self.get_word_frequencies('ham')

        # Calculate number of features in the training set
        vocabulary_spam = set(self.word_frequencies_spam.keys())
        vocabulary_ham = set(self.word_frequencies_ham.keys())
        self.n_vocabulary = len(vocabulary_spam.union(vocabulary_ham))

    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()
    
    def sms_classify(self, message, alpha=0.5):
        '''
        classifies a single message as spam or ham
        Takes in as input a new sms (w1, w2, ..., wn),
        performs the same data cleaning steps as in the training set,
        calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
        compares them and outcomes whether the message is spam or not.
        '''
        # Clean and tokenize message
        message = self.clean_text(message)
        message = message.split()

        # Calculate P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn)
        p_ham_given_message = self.p_ham
        p_spam_given_message = self.p_spam

        for word in message:
            # Calculate P(wi|Spam) with Laplace smoothing
            frequency_spam = self.word_frequencies_spam[word] if word in self.word_frequencies_spam else 0
            p_spam_given_message *= (frequency_spam + alpha) / (self.n_spam + alpha * self.n_vocabulary)

            # Calculate P(wi|Ham) with Laplace smoothing
            frequency_ham = self.word_frequencies_ham[word] if word in self.word_frequencies_ham else 0
            p_ham_given_message *= (frequency_ham + alpha) / (self.n_ham + alpha * self.n_vocabulary)

        # Determine class with highest conditional probability
        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'

    def classify_test(self):
        '''
        Calculate the accuracy of the algorithm on the test set and returns 
        the accuracy as a percentage.
        '''
        # Train naive bayes classifier
        self.train()

        # Classify each sample in the test data
        predictions = []
        for message in self.test_set['v2']:
            predictions.append(self.sms_classify(message))

        # Calculate test accuracy
        correct = predictions == self.test_set['v1']
        accuracy = sum(correct) / len(correct)

        return accuracy * 100

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()

    # Calculate test accuracy
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
