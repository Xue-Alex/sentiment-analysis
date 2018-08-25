from src.naive_bayes import MultinomialNaiveBayes
import unittest

temp = MultinomialNaiveBayes()
temp.learn()


class LearningCase(unittest.TestCase):

    def test_sanity(self):
        self.assertEqual(temp.predict(temp.tokenizer('bad bad bad bad ')),False)
        self.assertEqual(temp.predict (temp.tokenizer('good good excellent great')), True)







