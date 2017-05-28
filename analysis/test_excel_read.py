#!/usr/bin/env python

import csv
import sys
import numpy as np
import matplotlib as plt
import pandas as pd 

print ("===============")
table1 = pd.read_excel("excel_test_file.xlsx", sheetname = 'sheet1', header=3)
print (table1)

print ("===============")
table2 = pd.read_excel("excel_test_file.xlsx", sheetname = 'sheet2', header=3)
print (table2)

print ("===============")
table3_concat = pd.concat([table1, table2], ignore_index=False) 
print (table3_concat)


# ********************************************
# -- 엑셀 파일 쓰기(to_excel)
#    [xlsxwriter, xlrd] 패키지가 설치되어 있어야 함
# ********************************************
excelOutPath = "format_new.xlsx"    # .xls는 안되더라?
writer = pd.ExcelWriter(excelOutPath, engine="xlsxwriter")

table3_concat.to_excel(writer, sheet_name="Sheetall", index=True)
writer.save()

print ("===============")
table1 = pd.read_excel("excel_test_file.xlsx", sheetname = 'sheet1')
print (table1)

print ("===============")
table2 = pd.read_excel("excel_test_file.xlsx", sheetname = 'sheet1', header=3)
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
