import sys
import argparse
import numpy as np


def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)


p = argparse.ArgumentParser()
p.add_argument("-a", "--answerfile", help="answer file (ground truth)", type=argparse.FileType('r'),
               default='sample.answer')
p.add_argument("-r", "--resultfile", help="result file", type=argparse.FileType('r'),
               default='sample.result')
args = p.parse_args()

answer = read_data(args.answerfile)
predicted = read_data(args.resultfile)
lis = answer == predicted
flag = 0
for i in lis:
    if i:
        flag += 1

print(flag / len(lis))
