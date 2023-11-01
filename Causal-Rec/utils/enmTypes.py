from enum import Enum


class InputType(Enum):
    """Type of Models' input.
    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3
    USERWISE = 4

class LossType(Enum):

    BPR = 1
    UBPR = 2
    RELMF = 3
    EBPR = 4
    PDA = 5
    UPL = 6
    MFDU = 7
    DPR = 8
    BCE = 9



class EvalType(Enum):

    FULLSORT = 1
    NEG_SAMPLE_SORT = 2

class DataStatic(object):
    def __init__(self,name,user_num,item_num):
        self.name = name[0]
        self.user_num = user_num[0]
        self.item_num = item_num[0]
    def out(self):
        print(f'dataset name is {self.name}, user num is {self.user_num}, item num is {self.item_num}')
