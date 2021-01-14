import ioHelper
import userHelper
import os
from typing import *


def create_dict(name:str) -> Dict[str, int]:
    r""" create dict

    inputs: path
        -**path**: path of file

    outputs: data
        -**data**: a dictionary given the str retrun the index

    """

    holder = {}
    path = os.getcwd()
    vocab = ioHelper.read(path + '/Data/vocab.txt')
    sound = ioHelper.read(path + name)
    for key, value in zip(vocab, sound):
        split_list = value.split(' ')
        if len(split_list) > 1:
            holder[key] = int(split_list[0])
        else:
            holder[key] = int(value)
    return holder


def create_vowel_dict() -> Dict[str, int]:
    r""" create vowel dict

    inputs: None

    outputs: vowel
        -**vowel**: a dictionary given the str retrun the vowel

    """
    return create_dict('/Data/vowel.txt')


def create_tone_dict() -> Dict[str, int]:
    r""" create tone dict

    inputs: None

    outputs: tone
        -**tone**: a dictionary given the str retrun the tone

    """

    return create_dict('/Data/tone.txt')


# def main():
#     a = create_tone_dict()
#     return 0
#
#
# if __name__ == "__main__":
#     main()
