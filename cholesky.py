import numpy as np
import math
    
def cholesky(A):
    if np.size(A) == 1:
        return math.sqrt(A[0, 0])
    elif np.size(A, axis=0) == 2:
        i = np.size(A, axis=0) - 1
        a = A[0, 1]
        L = cholesky(A[:i, :i])
        l = a / L
        l_i = math.sqrt(A[i, i] - (l * l))
        L = np.array([L, 0])
        l = np.array([l, l_i])
        L = np.vstack((L, l))
        return L
    else:
        i = np.size(A, axis=0) - 1
        a = A[:i, i]
        L = cholesky(A[:i, :i])
        l = np.linalg.solve(L, a)
        l_i = math.sqrt(A[i, i] - np.dot(l, l.T))
        zeros = np.zeros((np.size(L, axis=0), 1))
        L = np.hstack((L, zeros))
        l = np.append(l, l_i)
        L = np.vstack((L, l))
        return L
    
def test():    
    A = np.array([[16, 4, 4, -4],
                  [4, 10, 4, 2],
                  [4, 4, 6, -2],
                  [-4, 2, -2, 4]])
              
    L = cholesky(A)

    print(L)
    print(np.dot(L, L.T))
    
#test()
