import math
import itertools
from scipy.stats import norm
import numpy as np
from typing import List, Tuple
from scipy.stats import beta

def generateClopperPearsonInterval(num: int, den: int) -> Tuple[float, float]:
    confidenceLevel = 0.68
    alpha = 1 - confidenceLevel

    if num == 0: lowerLimit = 0
    else: lowerLimit = beta.ppf(alpha / 2, num, den - num + 1)

    if num == den: upperLimit = 1
    else: upperLimit = beta.ppf(1 - alpha / 2, num + 1, den - num)

    return float(lowerLimit), float(upperLimit)

def stdError(avg1,d1,avg2,d2):
    stdError = math.sqrt( avg1*(1-avg1)/d1 + avg2*(1-avg2)/d2 )
    return stdError

# returns None when both n1 and n2 are 0
def zStat(n1,d1,n2,d2):
    avg1 = float(n1) / d1
    avg2 = float(n2) / d2

    err = stdError(avg1,d1,avg2,d2)
    if err != 0:
        return ( avg1 - avg2 )  / err
    else:
        return None

## Returns None when z_stat is None
def pValue(n1,d1,n2,d2):
    z_stat = zStat(n1,d1,n2,d2)
    if z_stat is not None:
        z_stat = abs(z_stat)
        return 1 - (norm.cdf(z_stat) - norm.cdf(-z_stat))
    else:
        return None


class ItemEfficiency:
    def __init__(self,name):
        self.name = name
        self.format = np.dtype([('source', 'U100'), ('numerator', 'i4'), ('denominator', 'i4'),('efficiency', 'f4')])
        self.data = np.array([],dtype=self.format)
        self.verbose = False
        self.__consistent = True
        self.consistencyMap = {}
        self.sourceIndex = {}
        self.discrepancy_cut = 0.1
        self.base_string = f"{self.name} has efficiency inconsistency > {self.discrepancy_cut*100}% in"
        

    def AddData(self,source,Num,Den):
        if source in self.data['source']:
            print(f"Data is present for source {source} item name {self.name}.\nData points not added")
        elif Den == 0:
            print(f"Parsed denominator is 0.\nData points not added")
        elif Num > Den:
            print(f"Parsed numerator {Num} > denominator {Den}.\nData points not added")
        else:
            self.data = np.append(self.data,np.array([(source,Num,Den,float(Num)/Den)],dtype=self.format))
    def ClearData(self):
        self.data = np.array([],dtype=self.format)

    def __sortData(self):
        idx = np.argsort(self.data, order=['source'])
        sorted_data = self.data[idx]
        for k in range(len(idx)):
            self.sourceIndex[k] = sorted_data['source'][k]
        return sorted_data

    def isConsistent(self):
        return self.__consistent

    def getConsistencyResults(self):
        return self.consistencyMap

    def getSourceIndex(self):
        return self.sourceIndex

    
    def checkConsistency(self):
        sortedData = self.__sortData()
        data_size = len(sortedData)
        ## all possible pairs 
        for idx1,idx2 in itertools.combinations(list(range(0,data_size)),2):
            lwL1,upL1 = generateClopperPearsonInterval(sortedData['numerator'][idx1],sortedData['denominator'][idx1])
            lwL2,upL2 = generateClopperPearsonInterval(sortedData['numerator'][idx2],sortedData['denominator'][idx2])
            avg1 = ( lwL1 + upL1) / 2
            avg2 = ( lwL2 + upL2) / 2
            
            if avg1 > avg2:
                if lwL1 - upL2 > self.discrepancy_cut:
                    consistent = False
                    self.__consistent = False
                else:
                    consistent = True
            elif avg2 > avg1:
                if lwL2 - upL1 > self.discrepancy_cut:
                    consistent = False
                    self.__consistent = False
                else:
                    consistent = True
            else:
                consistent = True
            self.consistencyMap[(idx1,idx2)] = consistent
            if self.verbose:
                print(f"Eff[{sortedData['source'][idx1]}]:{sortedData['efficiency'][idx1]}")
                print(f"Eff[{sortedData['source'][idx2]}]:{sortedData['efficiency'][idx2]}")
                print(f"pvalue: {p}")
                if consistent: print(f"\033[1;92mConsistent\033[1;0m")
                else: print(f"\033[1;31mInconsistent\033[1;0m")
                print()
    
    def GetInconsistencySummary(self):
        k = self.base_string
        for key,value in self.consistencyMap.items():
            if value == False:
                a,b = key
                f1 = self.sourceIndex[a]
                f2 = self.sourceIndex[b]
                k += f"\n\t{f1}\t{f2}"
        if k == self.base_string: return ""
        else: return k + "\n"

    def checkConsistency2(self):
        sortedData = self.__sortData()
        data_size = len(sortedData)
        ## all possible pairs 
        for idx1,idx2 in itertools.combinations(list(range(0,data_size)),2):
            p = pValue(sortedData['numerator'][idx1],
                       sortedData['denominator'][idx1],
                       sortedData['numerator'][idx2],
                       sortedData['denominator'][idx2])
            if p is None:
                continue
            ## If the pvalue < 0.001
            if p < 0.001 : 
                consistent = False
                self.__consistent = False
            else: consistent = True

            self.consistencyMap[(idx1,idx2)] = consistent

            if self.verbose:
                print(f"Eff[{sortedData['source'][idx1]}]:{sortedData['efficiency'][idx1]}")
                print(f"Eff[{sortedData['source'][idx2]}]:{sortedData['efficiency'][idx2]}")
                print(f"pvalue: {p}")
                if consistent: print(f"\033[1;92mConsistent\033[1;0m")
                else: print(f"\033[1;31mInconsistent\033[1;0m")
                print()