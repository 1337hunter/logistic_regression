#!/usr/bin/env python3

import  sys
import  csv
import pandas as pd

class Describe:
    def __init__(self, filename):
        self.filename = filename
        self.content = []

    def ReadFile(self):
        with open(self.filename, 'r') as file:
            coco = csv.DictReader(file)
            for row in coco:
                self.content += [row]

    def Describe(self):
        pass

    def __call__(self):
        self.ReadFile()


def main():
    best_class = Describe(sys.argv[1])
    best_class.ReadFile()
    best_class.Describe()

def CheckArgs():
    if len(sys.argv) != 2:
        print(f"Usage: {__file__} <dataset>")
        exit()

if __name__ == '__main__':
    CheckArgs()
    main()
