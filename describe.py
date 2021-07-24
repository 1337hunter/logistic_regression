#!/usr/bin/env python3

import  sys
import  csv
import  datetime
import  math
from tabulate import tabulate

def isfloat(val):
    try:
        val = float(val)
        if math.isnan(val):
            return False
        return True
    except:
        return False

class Describe:
    def __init__(self, filename):
        self.filename = filename
        self.content = []
        self.listed = {}
        self.mean = {}
        self.count = {}
        self.columns = []
        self.min = {}
        self.max = {}
        self.std = {}
        self.Q25 = {}
        self.Q50 = {}
        self.Q75 = {}

    def ReadFile(self):
        with open(self.filename, 'r') as file:
            coco = csv.DictReader(file)
            for row in coco:
                del row['Index']
                newrow = {}
                for k, v in row.items():
                    if isfloat(v):
                        newrow[k] = float(v)
                        if k not in self.listed.keys():
                            self.listed[k] = [float(v)]
                        else:
                            self.listed[k] += [float(v)]
                    elif k == 'Birthday':
                        split = v.split('-')
                        year, month, day = int(split[0]), int(split[1]), int(split[2])
                        newrow[k] = datetime.datetime(year, month, day, 0, 0).timestamp()
                        if k not in self.listed.keys():
                            self.listed[k] = [newrow[k]]
                        else:
                            self.listed[k] += [newrow[k]]
                self.content += [newrow]

    def FilterNumerics(self):
        for k, v in self.content[0].items():
            try:
                float(v)
                self.columns += [k]
                self.mean[k] = 0
                self.count[k] = 0
                self.std[k] = 0
                self.min[k] = 0
                self.max[k] = 0
            except:
                pass
    
    def GetCount(self):
        for x in self.content:
            for k, v in x.items():
                self.count[k] += 1

    def GetMean(self):
        for x in self.content:
            for k, v in x.items():
                self.mean[k] += v / self.count[k]

    def GetStd(self):
        for x in self.content:
            for k, v in x.items():
                self.std[k] += (v - self.mean[k]) ** 2 / self.count[k]
        for k, v in self.std.items():
            self.std[k] = math.sqrt(self.std[k])
    
    def GetQMinMax(self):
        for k in self.listed.keys():
            self.listed[k] = sorted(self.listed[k])
            if self.listed[k] != []:
                self.min[k] = self.listed[k][0]
                self.max[k] = self.listed[k][-1]
            else:
                continue
            L25 = (self.count[k] + 1) * 0.25
            L50 = (self.count[k] + 1) * 0.5
            L75 = (self.count[k] + 1) * 0.75
            try:
                P25 = self.listed[k][int(L25)] + (L25 - int(L25)) * (self.listed[k][int(L25) + 1] - self.listed[k][int(L25)])
                P50 = self.listed[k][int(L50)] + (L50 - int(L50)) * (self.listed[k][int(L50) + 1] - self.listed[k][int(L25)])
                P75 = self.listed[k][int(L75)] + (L75 - int(L75)) * (self.listed[k][int(L75) + 1] - self.listed[k][int(L25)])
            except:
                P25 = self.listed[k][0]
                P50 = self.listed[k][0]
                P75 = self.listed[k][0]
            self.Q25[k] = P25
            self.Q50[k] = P50
            self.Q75[k] = P75

    def Describe(self):
        self.GetCount()
        self.GetMean()
        self.GetStd()
        self.GetQMinMax()

    def Print(self):
        self.columns = sorted(self.columns)
        self.mean = {k: v for k, v in sorted(self.mean.items(), key=lambda item: item[0])}
        self.count = {k: v for k, v in sorted(self.count.items(), key=lambda item: item[0])}
        self.min = {k: v for k, v in sorted(self.min.items(), key=lambda item: item[0])}
        self.max = {k: v for k, v in sorted(self.max.items(), key=lambda item: item[0])}
        self.std = {k: v for k, v in sorted(self.std.items(), key=lambda item: item[0])}
        self.Q25 = {k: v for k, v in sorted(self.Q25.items(), key=lambda item: item[0])}
        self.Q50 = {k: v for k, v in sorted(self.Q50.items(), key=lambda item: item[0])}
        self.Q75 = {k: v for k, v in sorted(self.Q75.items(), key=lambda item: item[0])}
        columns = [''] + self.columns

        print(tabulate([
            ['Count'] + list(self.count.values()), 
            ['Mean'] +  list(self.mean.values()),
            ['Std'] + list(self.std.values()),
            ['Min'] + list(self.min.values()),
            ['25%'] + list(self.Q25.values()),
            ['50%'] + list(self.Q50.values()),
            ['75%'] + list(self.Q75.values()),
            ['Max'] + list(self.max.values())], headers=columns, tablefmt='plain', floatfmt=".6f"))

    def ConvertBirthday(self):
        start = datetime.datetime.fromtimestamp(0)
        self.mean['Birthday'] = datetime.datetime.fromtimestamp(self.mean['Birthday']).strftime('%Y-%m-%d')
        self.std['Birthday'] = (datetime.datetime.fromtimestamp(self.std['Birthday']) - start).days
        self.min['Birthday'] = datetime.datetime.fromtimestamp(self.min['Birthday']).strftime('%Y-%m-%d')
        self.max['Birthday'] = datetime.datetime.fromtimestamp(self.max['Birthday']).strftime('%Y-%m-%d')
        self.Q25['Birthday'] = datetime.datetime.fromtimestamp(self.Q25['Birthday']).strftime('%Y-%m-%d')
        self.Q50['Birthday'] = datetime.datetime.fromtimestamp(self.Q50['Birthday']).strftime('%Y-%m-%d')
        self.Q75['Birthday'] = datetime.datetime.fromtimestamp(self.Q75['Birthday']).strftime('%Y-%m-%d')
        pass

    def __call__(self):
        self.ReadFile()
        self.FilterNumerics()
        self.Describe()
        self.ConvertBirthday()
        self.Print()


def main():
    best_class = Describe(sys.argv[1])
    best_class()

def CheckArgs():
    if len(sys.argv) != 2:
        print(f"Usage: {__file__} <dataset>")
        exit()

if __name__ == '__main__':
    CheckArgs()
    main()
