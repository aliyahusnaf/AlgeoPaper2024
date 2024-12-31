import numpy as np 
def multiply_matrices(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def DotProduct(M,N):
    sum = 0
    for i in range (len(M)):
        sum = sum + M[i] * N[i]
    return sum


def transpose(matrix):
    if len(matrix) == 0 or len(matrix[0]) == 0:
        raise ValueError("Input matrix is empty or malformed.")
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed


def length(vector):
    return sum(x**2 for x in vector) ** 0.5

def normalize(col):
    norm = np.sqrt(np.dot(col, col))
    return [x / norm for x in col]


def sort_eigen(eigenvalues, eigenvectors):
    sortedind = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
    sorted_eigenvalues = [eigenvalues[i] for i in sortedind]
    sorted_eigenvectors = [[eigenvectors[j][i] for i in sortedind] for j in range(len(eigenvectors))]
    return sorted_eigenvalues, sorted_eigenvectors
    
def QR(matrix, iterations=10):
    n = len(matrix)
    A = np.array(matrix, dtype=np.float64)  
    eigenvectors = np.eye(n)  

    for _ in range(iterations):
        Q = []
        for i in range(n):
            col = A[:, i]
            for prev in Q:
                scale = np.dot(col, prev)
                col -= scale * prev
            norm = np.linalg.norm(col)
            if norm > 1e-10: 
                col = col / norm
            else:
                col = np.zeros_like(col)  
            Q.append(col)
        Q = np.array(Q).T 
        R = np.dot(Q.T, A)
        A = np.dot(R, Q) 
        eigenvectors = np.dot(eigenvectors, Q)

    eigenvalues = np.diag(A) 
    return eigenvalues, eigenvectors.tolist()



def svd(A):

    A = (A - np.mean(A, axis=0)) / (np.std(A, axis=0) + 1e-10)
    AAT = np.dot(A, transpose(A))
    AAT += np.eye(A.shape[0]) * 1e-6
    eigenvalues_AAT, eigenvectors_AAT = QR(AAT)
    eigenvalues_AAT, eigenvectors_AAT = sort_eigen(eigenvalues_AAT, eigenvectors_AAT)
    U = transpose(eigenvectors_AAT)
    return U


