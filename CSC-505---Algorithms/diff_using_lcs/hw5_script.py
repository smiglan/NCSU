# -*- coding: utf-8 -*-
import sys

def hash(filename1):
    hash_list = []
    with open(filename1,'r') as f:
        content = f.readlines()
    for i in range(len(content)):
        content[i] = content[i].rstrip()
    for x in content:
        hash = 0
        for i in x:
            hash = hash + ord(i)
        hash_list.append(hash%1024)
    return hash_list,content

def deriveLCSFromDictionaryOperations(A, B, directions, i, j ):
    path = []
    sign = []
    operations = []

    while True:
        if i <= 0 and j <= 0:
            break
        if directions[i, j] == 'D':
            path.append(A[i - 1])
            sign.append('  ')
            operations.append(A[i - 1])
            i -= 1
            j -= 1
        elif directions[i, j] == 'U':
            sign.append('+ ')
            operations.append(B[j - 1])
            j -= 1
        elif directions[i, j] == 'L':
            sign.append('- ')
            operations.append(A[i - 1])
            i -= 1
        elif directions[i, j] == 'E':
            if i == 0:
                sign.append('+ ')
                operations.append(B[j - 1])
                j -= 1
            elif j == 0:
                sign.append('- ')
                operations.append(A[j - 1])
                i -= 1
    return path,sign,operations


def fillDictionaryWithOperations(A, B, directions, lcsLen):
    for i in range(len(A) + 1):
        for j in range(len(B) + 1):
            if i == 0 or j == 0:
                lcsLen[i, j] = 0
                directions[i, j] = "E"
            elif A[i - 1] == B[j - 1]:
                lcsLen[i, j] = lcsLen[i - 1, j - 1] + 1
                directions[i, j] = "D"
            else:
                if lcsLen[i - 1, j] >= lcsLen[i, j - 1]:
                    lcsLen[i, j] = lcsLen[i - 1, j]
                    directions[i, j] = "L"
                else:
                    lcsLen[i, j] = lcsLen[i, j - 1]
                    directions[i, j] = "U"
                    
inputf = sys.argv[1]
filename1 = sys.argv[2]

if inputf == 'hashfile':
    for i in hash(filename1)[0]: print(i)
else:
    filename2 = sys.argv[3]  
    A,A_content = hash(filename1)
    B,B_content = hash(filename2)
    A = list(A)
    B = list(B)

    lcsLen = {}
    directions = {}
    fillDictionaryWithOperations(A, B, directions, lcsLen)

    i = len(A)
    j = len(B)
    path,sign,operations = deriveLCSFromDictionaryOperations(A, B, directions, i, j)
    path.reverse()
    sign.reverse()
    operations.reverse()

    file1_index = []
    file2_index = []
    for p in path:
      file1_index.append(A.index(int(p))+1)
      file2_index.append(B.index(int(p))+1)

    for i,o in enumerate(operations):
      if o in A:
        operations[i] = A_content[A.index(o)]
      elif o in B:
        operations[i] = B_content[B.index(o)]  
if inputf == 'diff':
    print(*file1_index, sep=" ")
    print(*file2_index, sep=" ")
elif inputf == 'diffprint':
    for a in range(len(operations)): print(sign[a],operations[a])
