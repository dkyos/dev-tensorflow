#!/usr/bin/env python

import difflib as dl

#a = file('file').read()
#b = file('file1').read()

a = "I'd like an apple"

#b = "Apple is my favorite fruit"
b = "An apple a day keeps the doctor away"

sim = dl.get_close_matches

s = 0
wa = a.split()
wb = b.split()

for i in wa:
    if sim(i, wb):
        s += 1

n = float(s) / float(len(wa))
print ('%d%% similarity' % int(n * 100))
