#!/usr/bin/env python

import csv
import sys
import numpy as np
import matplotlib as plt
import pandas as pd 

print ("===============")
table1 = pd.read_excel("excel_test_file.xlsx", sheetname = 'tab11')
print (table1)

print ("===============")
table2 = pd.read_excel("excel_test_file.xlsx", sheetname = 'tab11', header=3)
print (table2)
print ("--------")
print (table2.index)
print (table2.ix['row1'])
print (table2.ix[1:])
print ("--------")
print (table2.columns)
print (table2['column2'])
print (table2[ [ 'column2','column1'] ])

print ("===============")
table2 = table2.drop('note', 1)
print (table2)

print ("===============")
print (table2.index)
print (table2['row2':'row3'])
