import string
from random import shuffle
from pymongo import MongoClient


class DbConnect():

    def __init__(self):
        self._db_col = MongoClient().test_database.test_collection
        self.queries = []

    def retrieve(self):
        for document in self._db_col.find():
            self.queries.append(document['text'])
        shuffle(self.queries)
        return self.data_split(self.queries)

    def data_split(self, lines):
        num_lines = len(lines)
        line_count = 0
        train_size = int(0.8 * num_lines)
        training = []
        testing = []
        for line in lines:
            if line_count <= train_size:
                training.append(line)
            else:
                testing.append(line)
            line_count += 1
        return training, testing


class MultinomialNaiveBayes:

    def __init__(self):
        self.datasets = []
        self.training_set = []
        self.testing_set = []
        self.neg_occ = {}
        self.pos_occ = {}
        self.unique_words = {}
        self.training_set, self.testing_set = DbConnect().retrieve()
        self.num_bad = 0
        self.num_good = 0
        self.neg_words = 0
        self.pos_words = 0

    def tokenizer(self, query):
        punctuation = set(string.punctuation + '-')
        for c in punctuation:
            query = query.replace(c, '')
        query = query.strip('\n')
        query = query.lower()
        return query.split(' ')

    def clean_rate(self, rate):
        rate = rate.strip('\n')
        return rate

    def bag_of_words(self, query, rate):
        bag = self.tokenizer(query)
        for word in bag:
            if word not in self.pos_occ:
                self.pos_occ[word] = 0
                self.unique_words[word] = 1
            if word not in self.neg_occ:
                self.neg_occ[word] = 0
                self.unique_words[word] = 1
            if rate == '1':
                self.pos_occ[word] += 1
            if rate == '0':
                self.neg_occ[word] += 1

    def learn(self):
        for c in self.training_set:
            q = c.split('\t')
            q[1] = self.clean_rate(q[1])
            if q[1] == '0':
                self.num_bad += 1
            else:
                self.num_good += 1
            self.bag_of_words(q[0], q[1])
        self.neg_words = sum(self.neg_occ.values())
        self.pos_words = sum(self.pos_occ.values())

    def predict(self, query):
        prob_good = 1.0
        prob_bad = 1.0
        for word in query:
            if word in self.pos_occ:
                prob_good *= (self.pos_occ[word] + 1) / (self.pos_words + sum(self.unique_words.values()))
            else:
                prob_good *= 1 / (self.pos_words + sum(self.unique_words.values()))
            if word in self.neg_occ:
                prob_bad *= (self.neg_occ[word] + 1) / (self.neg_words + sum(self.unique_words.values()))
            else:
                prob_bad *= 1 / (self.neg_words + sum(self.unique_words.values()))

        return prob_good * (self.num_good / (self.num_good + self.num_bad)) > prob_bad * (
                self.num_bad / (self.num_good + self.num_bad))

    def evaluate(self):
        correct = 0
        total = 0
        errors = []
        for c in self.testing_set:
            total += 1
            q = c.split('\t')
            prediction = self.predict(self.tokenizer(q[0]))
            rate = self.clean_rate(q[1])
            if prediction and rate == '1':
                correct += 1
            elif not prediction and rate == '0':
                correct += 1
            else:
                errors.append(c)
        return correct / total


temp = MultinomialNaiveBayes()
temp.learn()
print(temp.evaluate())
