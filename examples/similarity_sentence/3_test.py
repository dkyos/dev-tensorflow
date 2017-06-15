#!/usr/bin/env python


from sklearn.feature_extraction.text import TfidfVectorizer

#documents = [open(f) for f in text_files]
#tfidf = TfidfVectorizer().fit_transform(documents)
## no need to normalize, since Vectorizer will return normalized tf-idf
#pairwise_similarity = tfidf * tfidf.T

#or, if the documents are plain strings,
vect = TfidfVectorizer(min_df=1);
tfidf = vect.fit_transform([
    "I'd like an apple",
    "An apple a day keeps the doctor away",
    "Apple is my favorite fruit",
    "Never compare an apple to an orange",
    "I prefer scikit-learn to Orange"
]);

print ( (tfidf * tfidf.T).A )
