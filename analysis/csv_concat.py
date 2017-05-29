#!/usr/bin/env python

import csv
import sys
import numpy as np
import matplotlib as plt
import pandas as pd 

list_iter = iter(["02_20170516_A.csv","02_20170516_G.csv"])

df1 = pd.DataFrame()
df_concat = pd.DataFrame()

for iter in list_iter:
    print(iter)

    print ("===============")
    df1 = pd.read_csv(iter, sep='|')
    print (df1.head(1))

    print ("===============")
    df_concat = pd.concat([df_concat, df1], ignore_index=False) 
    print (df_concat.head(2))

    # get the columns from one of them
    all_columns = df1.columns

    # finally, re-index the new dataframe using the original column index
    df_concat = df_concat.ix[:, all_columns]


print ("===============")
output = "concat_result.csv"
df_concat.to_csv(output, index=False, sep='|')



