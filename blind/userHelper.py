#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
import numpy as np
import os
import json
import argparse
import ioHelper
from typing import *


#function return max n element in accending order
#output: List of turple in the form of (data, position)
def max_n(arr:np.ndarray, n:int) -> List[Tuple[np.ndarray, int]]:
    r""" top n elment

    inputs: array, n
        -**array**: a list of score
        -**n**: index to keep

    outputs: index
        -**index**: list of tuple [score, index]

    """

    indices = arr.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return [[arr[i], i[0]] for i in indices]


#convert data to number
def convert_to_id(input:List[str], vocab:Dict[str,int]) -> List[int]:
    r""" convert str to id

    inputs: data, dict
        -**data**: a list of string data
        -**dict**: a dictionary to map char to int

    outputs: int_data
        -**index**: list of int data

    """

    store = []
    for i in input:
        store.append(vocab[i])
    return store


#name of tune library
def loadTitle() -> List[List[str]]:
    r""" load tune title

    inputs: None

    outputs: index
        -**index**: list of tune str

    """

    title_list = readFile("title_lib.txt")
    return title_list


def flatten(seq:List, container = None) -> List:
    r""" load tune title

    inputs: data
        -**data**: list of list of data

    outputs: flat_data
        -**flat_data**: data in one list format

    """

    if container == None:
        container = []
    for s in seq:
        if hasattr(s, '__iter__'):
            flatten(s, container)
        else:
            container.append(s)
    return container


def reverse(vocab:List[str]) -> Dict[str,int]:
    r""" reverse data

    inputs: data
        -**data**: list of data

    outputs: dict
        -**dict**: a dictionary of build by data

    """

    store = {}
    count = 0
    for i in vocab:
        store[i] = count
        count += 1
    return store
