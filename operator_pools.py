import openfermion
import numpy as np
import copy as cp
import re
import scipy

import random

from openfermion import *



class OperatorPool:

    def __init__(self,ListOfStrings=['YIIIIIII']):


        self.n_spin_orb=len(ListOfStrings[0])
        self.generate_Pool_Operators(ListOfStrings)



    def generate_Pool_Operators(self,ListOfStrings):
        self.fermi_ops=[]
        for string in ListOfStrings:
            formated_str=''
            for letter in range(len(string)):
                if string[letter]!='I':
                    formated_str+=string[letter]+str(letter)+' '
            self.fermi_ops.append(QubitOperator(formated_str,1j))
        self.n_ops = len(self.fermi_ops)

        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits = self.n_spin_orb))



