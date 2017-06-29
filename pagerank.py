#!/usr/bin/env python
import numpy as np 
import pandas as pd
import scipy 
import scipy.sparse as ss 
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy.sparse import coo_matrix
import time
import csv
from sklearn import preprocessing  

# power method for computing top singular value and corresponding right singular vector for a given CSR matrix
# power method on A returns the eigenvector of A
# power method on AtA returns the eigenvector of AtA
# the eigenvector of AtA is the right singular vector of A
def power_method( sparse_matrix_in_csr_format, number_of_iterations ):
  start = time.time()
  dim = sparse_matrix_in_csr_format.shape[1]
  v = np.random.rand(dim, 1) # vertical vector

  for _ in range(number_of_iterations):

    v = sparse_matrix_in_csr_format @ v # Av
    v = sparse_matrix_in_csr_format.transpose() @ v # AtAv
    v = v / np.linalg.norm(v)

  quality = quality_of_solution( sparse_matrix_in_csr_format, v )
  qualityfrob = quality / linalg.norm(sparse_matrix_in_csr_format)

  top_right_singular_vector = v.transpose()[0] # we'll return this 1d array

  vtA = v.transpose() * sparse_matrix_in_csr_format
  vtAv = vtA @ v 
  top_singular_value = vtAv[0][0]
  end = time.time()
  print(number_of_iterations, "iterations of power method took", (end-start)*1000, "ms")
  print("Quality of solution:", quality)
  print("Quality of solution (dividing by frobenius norm of A):", qualityfrob, "\n")
  return top_right_singular_vector, top_singular_value 

def quality_of_solution( sparsemat, vec ): # vec should be vertical
  # quality is vtAtAv
  vtAt = vec.transpose() * sparsemat.transpose() # quality = vtAt
  vtAtA = vtAt * sparsemat
  vtAtAv = vtAtA @ vec 
  q = vtAtAv[0][0]
  return q

def pagerank( P ):
  # start from an initial horizontal vector with unit sum
  dim = P.shape[1]
  x = np.random.rand(1, dim)
  x = preprocessing.normalize(x, norm = 'l1') # now x has unit sum
  # verify with:
  # x_sum = np.sum(x, axis=1) ; should give 1
  iterations = 150
  for _ in range(iterations):
    x = P.dot(x.T)
    x = x.T

  return x[0] # pagerank vector, which is a probability distribution

def main():
  
  data = pd.read_table('enwiki-2013.txt.gz', sep = ' ', dtype = np.int32, 
                      skiprows = 4, engine = "c", names = ("from", "to"))
  rows = data['from'] # type is <class 'pandas.core.series.Series'>
  cols = data['to'] 
  ones = np.ones(len(rows), np.int32) # type is <class 'numpy.ndarray'>
  n = max(rows)+1
  coomatrix = ss.coo_matrix((ones, (rows, cols)), shape = (n, n)) # type is <class 'scipy.sparse.coo.coo_matrix'>
  csrmatrix = coomatrix.tocsr() # <4206289x4206289 sparse matrix of type '<class 'numpy.int32'>' with 101311613 stored elements in Compressed Sparse Row format>
  # max(rows) = 4206288 ; max(cols) = 4206287
  
  # for each trial, report runtime and quality of solution
  num_iterations = 1
  top_right_singular_vector, top_singular_value = power_method( csrmatrix, num_iterations)
  num_iterations = 3
  top_right_singular_vector, top_singular_value = power_method( csrmatrix, num_iterations)
  num_iterations = 5
  top_right_singular_vector, top_singular_value = power_method( csrmatrix, num_iterations)
  num_iterations = 10
  top_right_singular_vector, top_singular_value = power_method( csrmatrix, num_iterations)
  num_iterations = 20
  top_right_singular_vector, top_singular_value = power_method( csrmatrix, num_iterations)
  
  print("\nVector of authority scores (top right singular vector):")
  print(top_right_singular_vector)
  print("\nTop singular value:")
  print(top_singular_value)

  csrmatrixf64 = csrmatrix.asfptype()
  
  start = time.time()
  svd_vecs = scipy.sparse.linalg.svds(csrmatrixf64, k = 1)
  end = time.time()
  svd_vec = svd_vecs[2] # horizontal vec
  print("\nsvd_vec:", svd_vec[0])
  print("\nsvd function took", (end-start)*1000, "ms")
  svd_quality = quality_of_solution(csrmatrix, svd_vec.transpose())
  print("\nQuality of solution (when computing v using svd, and not dividing by frobenius norm):", svd_quality)
  svd_quality = svd_quality / linalg.norm(csrmatrix)
  print("\nQuality of solution (when computing v using svd, and dividing by frobenius norm):", svd_quality, "\n")
  
  norm_auth_scores = top_right_singular_vector
  top5auth_indices = np.argsort(-top_right_singular_vector)[:5] # get indicies of the largest 5 values (unsorted)
  # top5auth_indices = top5auth_indices[np.argsort(top_right_singular_vector[top5auth_indices])]
  top5auth_scores = top_right_singular_vector[top5auth_indices]
  print("Indices of the best 5 authority pages:", top5auth_indices, "\nScores of the best 5 authority pages:", top5auth_scores, "\n")
  
  hub_scores = csrmatrix @ top_right_singular_vector
  hub_scores = hub_scores / np.linalg.norm(hub_scores)  # normalize
  top5hub_indices = np.argsort(-hub_scores)[:5] # get the 5 best pages
  # top5hub_indices = top5hub_indices[np.argsort(hub_scores[top5hub_indices])]  # sort 5 from best to worst
  top5hub_scores = hub_scores[top5hub_indices]
  print("Indices of the best 5 hub pages:", top5hub_indices, "\nScores of the best 5 hub pages:", top5hub_scores, "\n")
  
  csrmatrixf64 = csrmatrix.asfptype()
  P = preprocessing.normalize(csrmatrixf64, norm = 'l1') # rows have unit sum
  # to verify:
  # A_row_sum = np.sum(P, axis = 1)

  PRscores = pagerank(P)
  top5pr_indices = np.argsort(-PRscores)[:5]
  top5pr_scores = PRscores[top5pr_indices]
  print("Indices of the best 5 pagerank pages:", top5pr_indices, "\nScores of the best 5 pagerank pages:", top5pr_scores, "\n")

if __name__ == '__main__':
  main()