# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:07:51 2018

@author: MRVN
"""
""" Machine Learning TUD 2018, Ex.3
 
Decision Tree Learning:
Based on american census data you want to predict two classes of income of people:
>50K$, <=50K$.

We do not use continuous attributes for this first decision tree task.
"""
import numpy as np
__author__ = 'Benjamin Guthier'

from math import log

def openfile(path, fname):
    """opens the file at path+fname and returns a list of examples and attribute values.
    examples are returned as a list with one entry per example. Each entry then
    is a list of attribute values, one of them being the class label. The returned list attr
    contains one entry per attribute. Each entry is a list of possible values or an empty list
    for numeric attributes.
    """
    datafile = open(path + fname, "r")
    examples = []
    for line in datafile:
        line = line.strip()
        line = line.strip('.')
        # ignore empty lines. comments are marked with a |
        if len(line) == 0 or line[0] == '|':
            continue
        ex = [x.strip() for x in line.split(",")]
        examples.append(ex)

    attr = []
    for i in range(len(examples[0])):
        values = list({x[i] for x in examples}) # set of all different attribute values
        if values[0].isdigit():  # if the first value is a digit, assume all are numeric
            attr.append([])
        else:
            attr.append(values)
        
    return examples, attr


def calc_entropy(examples, cls_index, attr):
    """calculates the entropy over all examples. The index of the class label in the example
    is given by cls_index. Can also be the index to an attribute.
    """
    result = 0
    for i in attr[cls_index]: 
        print(i)
        print(attr[cls_index])
        pi = 0
        for j in examples: 
            
            #print(j)
            #print(examples[cls_index])
            #print(i)
            if(j[cls_index] == i):
                #print("dsfjljf")
                pi=pi+1
        if(pi==0):
            continue
        print(pi)
        pi = pi / len(examples)
        print("Anzahl: "+ str(len(examples)))
        result = result + ((-pi)*np.log2(pi))
    print(result)
    return result


def calc_ig(examples, attr_index, cls_index):
    """Calculates the information gain over all examples for a specific attribute. The
    class index must be specified.
    
    uses calc_entropy
    """
    
    return result


examples, attr = openfile(path='', fname='adult.data.txt')
#print(attr)
calc_entropy(examples, 9, attr)