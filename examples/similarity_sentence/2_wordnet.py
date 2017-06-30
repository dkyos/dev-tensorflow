#!/usr/bin/env python

# http://nlpforhackers.io/wordnet-sentence-similarity/

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


def path_similarity_won(synset, ss):
    a = synset.path_similarity(ss);
    if a is not None:
        ret = a;
    else:
        #ret = float('-inf')
        ret = 0
    return ret;

""" Convert between a Penn Treebank tag to a simplified Wordnet tag """;
def penn_to_wn(tag):
    if tag.startswith('N'):
        return 'n';

    if tag.startswith('V'):
        return 'v';

    if tag.startswith('J'):
        return 'a';

    if tag.startswith('R'):
        return 'r';

    return None;

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None;

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None;

""" compute the sentence similarity using Wordnet """;
def sentence_similarity(sentence1, sentence2):
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([path_similarity_won(synset, ss) for ss in synsets2])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
        count += 1

    # Average the values
    score /= count
    return score

sentences = [
    "Red alcoholic drink.",
    "White wine",
    "a type of wine made from dark-colored ",
    "Freshapple juice.",
    "Fresh orange juice.",
    "Dogs are awesome.",
    "Some gorgeous creatures are felines.",
    "Dolphins are swimming mammals.",
    "Cats are beautiful animals.",
]

focus_sentence = "Red wine"

print ("="*82)
#for sentence in sentences:
#    print ("Similarity(\"%s\", \"%s\") = %f" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence)))
#    print ("Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence)))
#    print 

# Similarity("Cats are beautiful animals.", "Dogs are awesome.") = 0.511111111111
# Similarity("Dogs are awesome.", "Cats are beautiful animals.") = 0.666666666667

# Similarity("Cats are beautiful animals.", "Some gorgeous creatures are felines.") = 0.833333333333
# Similarity("Some gorgeous creatures are felines.", "Cats are beautiful animals.") = 0.833333333333

# Similarity("Cats are beautiful animals.", "Dolphins are swimming mammals.") = 0.483333333333
# Similarity("Dolphins are swimming mammals.", "Cats are beautiful animals.") = 0.4

# Similarity("Cats are beautiful animals.", "Cats are beautiful animals.") = 1.0
# Similarity("Cats are beautiful animals.", "Cats are beautiful animals.") = 1.0

'''
print ("="*82)
print ("Building a symmetric similarity function")

""" compute the symmetric sentence similarity using Wordnet """;
def symmetric_sentence_similarity(sentence1, sentence2):
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 
         
for sentence in sentences:
    print ("SymmetricSimilarity(\"%s\", \"%s\") = %s" % (
        focus_sentence, 
        sentence, 
        symmetric_sentence_similarity(focus_sentence, sentence)
    ))
    print ("SymmetricSimilarity(\"%s\", \"%s\") = %s" % (
        sentence, 
        focus_sentence, 
        symmetric_sentence_similarity(sentence, focus_sentence)
    ))
    print 
'''

# SymmetricSimilarity("Cats are beautiful animals.", "Dogs are awesome.") = 0.588888888889
# SymmetricSimilarity("Dogs are awesome.", "Cats are beautiful animals.") = 0.588888888889
                           
# SymmetricSimilarity("Cats are beautiful animals.", "Some gorgeous creatures are felines.") = 0.833333333333
# SymmetricSimilarity("Some gorgeous creatures are felines.", "Cats are beautiful animals.") = 0.833333333333
                            
# SymmetricSimilarity("Cats are beautiful animals.", "Dolphins are swimming mammals.") = 0.441666666667
# SymmetricSimilarity("Dolphins are swimming mammals.", "Cats are beautiful animals.") = 0.441666666667
                             
# SymmetricSimilarity("Cats are beautiful animals.", "Cats are beautiful animals.") = 1.0
# SymmetricSimilarity("Cats are beautiful animals.", "Cats are beautiful animals.") = 1.0
                              
                               

