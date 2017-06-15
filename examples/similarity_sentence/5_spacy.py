#!/usr/bin/env python

import spacy
nlp = spacy.load('en')
doc1 = nlp(u'I would like an apple')
doc2 = nlp(u'Apple is not my favorite fruit')
doc3 = nlp(u'An apple a day keeps the doctor away')

print (doc1.similarity(doc2)) 
print (doc1.similarity(doc3)) 
