import json
from typing import *


def load(path:str) -> List[List[List[int]]]:
    r""" load json

    inputs: path
        -**path**: path of file

    outputs: data
       -**data**: json data

    """

    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump(data:List[List[List[int]]], path:str) -> None:
    r""" dump json

    inputs: path, data
        -**path**: path of file
        -**data**: json data

    outputs: None

    """

    with open(path, 'w') as f:
        json.dump(data, f, sort_keys = True)


# simple read file function, can have variation
def read(path:str) -> List[List[List[str]]]:
    r""" read

    inputs: path
        -**path**: path of file

    outputs: data
        -**data**: list of data

    """

    holder = []
    with open(path, 'r') as f:
        for lines in f:
            line = lines.replace("\n", "")
            holder.append(line)
    return holder

