
# Imports
import seaborn as sns

labels = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4,
              'environnement': 5,
              'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10,
              'education': 11,
              'catastrophes': 12, 'international': 13}

cf_matrix = [[134,1,5,2,3,5,0,1,0,2,0,0,2,3],
 [1,88,3,0,2,2,2,0,7,2,4,1,1,0],
 [ 31 ,17, 533,4,58,16,38,2,2 , 11 , 32,6,11,20],
 [0,1,0,38,2,1,0,0,0,0,1,0,0,0],
 [3,2,38,6, 326,20,6,0,0,0,0,0,3,4],
 [2,0,3,3,10, 129,1,0,0,0,1,0,8,2],
 [3,2,47,2,28,8 ,648,2,10,17,20,5,5,31],
 [0,3,0,0,1,1,1 ,234,1,1,1,0,1,2],
 [0,2,1,0,1,0,3,1 ,132,1,3,0,1,3],
 [0,0,8,0,8,1,13,3,1 ,184,18,0,1,7],
 [4,5,10,4,3,2,11,1,1,19 ,231,3,16,12],
 [1,0,4,0,0,0,1,1,0,0,2,93,0,0],
 [0,1,12,1,4,14,2,3,0,0,15,1,477,1],
 [0,5,12,3,14,3,25,1,4,11,10,1,8,607]]

#gin 512 2
cf_matrix = [[249, 2, 8, 2,25,23,10, 7, 1, 0, 1, 1, 0, 0],
 [ 16,38, 8, 0, 4, 4, 9, 1, 0, 0, 1, 0, 0, 0],
 [ 11, 8,78, 0, 9, 1, 7, 9, 2, 2, 1, 0, 0, 0],
 [1, 0, 4,50, 1, 4, 4, 9, 0, 0, 0, 0, 0, 0],
 [ 34, 5,11, 2 ,106, 5,17, 3, 0, 1, 1, 0, 0, 0],
 [ 29, 1, 0, 2, 7,92, 6, 3, 0, 0, 1, 0, 0, 0],
 [6, 5, 4, 1,14, 8 ,175, 4, 1, 2, 0, 0, 1, 0],
 [8, 1, 7, 3, 5, 4, 9, 140, 0, 2, 0, 0, 1, 0],
 [ 10, 0, 2, 0, 2, 1, 3, 1,24, 3, 0, 0, 0, 0],
 [1, 1, 1, 0, 4, 1, 5, 0, 1,34, 0, 0, 1, 0],
 [ 15, 1, 3, 4, 1, 2, 1, 2, 0, 1,26, 1, 0, 0],
 [7, 0, 3, 1, 1, 0, 0, 0, 0, 0, 1,16, 1, 0],
 [1, 2, 2, 0, 1, 0, 2, 3, 0, 0, 0, 0,44, 0],
 [1, 1, 0, 4, 0, 2, 0, 3, 0, 0, 0, 0, 2, 1]]

cf_matrix = [[ 32,   3,   3,   5,   0,   5,   4,   3,   0,   0, 102,   1,   0,
          0],
       [ 10,   1,   7,   1,   9,   7,   2,   2,  63,   6,   2,   0,   3,
          0],
       [541,   6,  29,   5,  62,  58,  32,  21,  10,   2,  10,   2,   2,
          1],
       [  8,   0,   0,   2,   2,   6,   0,   6,   3,   0,   1,   0,   0,
         15],
       [ 68,   1,   5,  14,  31, 264,  12,   8,   1,   0,   2,   0,   0,
          2],
       [ 12,   2,   1, 107,   3,   9,   1,  17,   0,   0,   7,   0,   0,
          0],
       [ 93,  22,  53,   4, 512,  18,  88,  19,   3,   6,   3,   2,   5,
          0],
       [ 10,   2,   4,   3,   3,   2,   3,   3,   4,   0,   1,   0, 211,
          0],
       [ 10,   0,  10,   1,  12,   1,  14,   2,   5,  91,   1,   0,   1,
          0],
       [ 19, 126,  28,   1,  40,   9,  14,   2,   0,   1,   1,   0,   3,
          0],
       [ 25,  12, 225,   0,  20,   2,  13,  13,   4,   1,   4,   3,   0,
          0],
       [ 18,   1,   5,   0,   5,   4,   3,   0,   0,   0,   2,  63,   1,
          0],
       [ 19,   1,  50,  17,  13,   1,  12, 412,   1,   0,   3,   1,   1,
          0],
       [ 23,  10,  45,   4,  48,  10, 543,   9,   3,   3,   1,   1,   4,
          0]]

import matplotlib.pyplot as plt

sns.heatmap(cf_matrix,cmap="Blues", annot=True,xticklabels=labels.keys(),yticklabels=labels.keys(), fmt="d")
plt.show()