'''import pandas as pd
import argparse

#Want to examine file MMDU.tsv

df = pd.read_csv("MMDU.tsv", sep="\t")
print(df.columns)
#print(df.iloc[0].to_dict())
print(df.iloc[0]["question"])'''

import pandas as pd
import ast

df = pd.read_csv("MMDU.tsv", sep="\t")

print(df.iloc[0]["question"])
print(type(df.iloc[0]["question"]))

q = ast.literal_eval(df.iloc[0]["question"])
print(type(q), len(q))
print(q[0])

print(df.iloc[0]["image_path"])
print(df.iloc[0]["answer"])