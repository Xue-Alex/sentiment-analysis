import string
import click
import os
from random import shuffle
from pymongo import MongoClient
from sw import stops


def data_split(lines):
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


class DbConnect:

    def __init__(self):
        self._db_col = MongoClient().test_database.test_collection
        self.queries = []

    def retrieve(self):
        for document in self._db_col.find():
            self.queries.append(document['text'])
        shuffle(self.queries)
        return data_split(self.queries)


def tokenizer(query):
    punctuation = set(string.punctuation + '-')
    for c in punctuation:
        query = query.replace(c, '')
    query = query.strip('\n').lower().split(' ')
    query[:] = [x for x in query if x not in stops() and x != '']
    return query


def get_data():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(cur_dir + "/../")
    resources_path = os.path.abspath(parent_dir + '/data/')
    training_set = []
    testing_set = []
    for files in os.listdir(resources_path):
        f = open(os.path.join(resources_path + f'\{files}'))
        temp_train, temp_test = data_split(list(f))
        training_set.extend(temp_train)
        testing_set.extend(temp_test)
    return training_set, testing_set


class MultinomialNaiveBayes:

    def __init__(self, db):
        self.datasets = []
        self.training_set = []
        self.testing_set = []
        self.neg_occ = {}
        self.pos_occ = {}
        self.unique_words = {}
        if db:
            self.training_set, self.testing_set = DbConnect().retrieve()
        else:
            self.training_set, self.testing_set = get_data()
        self.num_bad = 0
        self.num_good = 0
        self.neg_words = 0
        self.pos_words = 0

    def clean_rate(self, rate):
        rate = rate.strip('\n')
        return rate

    def bag_of_words(self, query, rate):
        bag = tokenizer(query)
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

    def n_grams(self, corp, rate, n = 3):
        bag = ' '.join(tokenizer(corp))
        for i in range(len(bag) - n + 1):
            gram = ''.join(bag[i:i + n])
            if gram not in self.pos_occ:
                self.pos_occ[gram] = 0
                self.unique_words[gram] = 1
            if gram not in self.neg_occ:
                self.neg_occ[gram] = 0
                self.unique_words[gram] = 1
            if rate == '1':
                self.pos_occ[gram] += 1
            if rate == '0':
                self.neg_occ[gram] += 1

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
        query = tokenizer(query)
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
            prediction = self.predict(q[0])
            rate = self.clean_rate(q[1])
            if prediction and rate == '1':
                correct += 1
            elif not prediction and rate == '0':
                correct += 1
            else:
                errors.append(c)
        return correct / total


@click.command()
@click.option('--db/--no-db', default = True)
def train(db):
    temp = MultinomialNaiveBayes(db)
    temp.learn()
    print(temp.evaluate())


if __name__ == '__main__':
    train()

