from pymongo import MongoClient
import os

client = MongoClient()

db = client.test_database

collection = db.test_collection

collection.deleteMany({})

for files in os.listdir('./data/'):
    f = open('./data/' + files, 'r')
    for line in f:
        collection.insert_one({'text': line})
