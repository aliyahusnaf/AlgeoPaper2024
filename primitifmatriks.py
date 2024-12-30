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


# def QR(matrix, iterations=10):
#     n = len(matrix)
#     A = [row[:] for row in matrix] # Matriks salinan
#     eigenvectors = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Matriks identitas

#     for _ in range(iterations):
#         Q = []
#         for i in range(n):
#             col = [A[j][i] for j in range(n)]
#             for prev in Q: # kolom sblmnya di Q
#                 scale = np.dot(col, prev)
#                 col = [col[k] - scale * prev[k] for k in range(n)]
#             Q.append(normalize(col))  
#         Q_T = transpose(Q)
#         R = np.dot(Q_T, A)
#         A = np.dot(Q_T, R)
#         eigenvectors = np.dot(eigenvectors, Q)
#     eigenvalues = [A[i][i] for i in range(n)]
#     return eigenvalues, eigenvectors

# def QR(matrix, iterations=10):
#     n = len(matrix)
#     A = [row[:] for row in matrix]  # Salinan matriks
#     eigenvectors = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Matriks identitas

#     for _ in range(iterations):
#         Q = []
#         for i in range(n):
#             col = [A[j][i] for j in range(n)]
#             for prev in Q:  # Kolom sebelumnya di Q
#                 scale = np.dot(col, prev)
#                 col = [col[k] - scale * prev[k] for k in range(n)]
#             norm = np.sqrt(np.dot(col, col))  # Normalisasi
#             col = [x / norm for x in col]
#             Q.append(col)

#         Q_T = np.array(transpose(Q)) # Transposisi menggunakan NumPy
#         R = np.dot(Q_T, A)  # Hitung R
#         A = np.dot(Q_T, R)  # Perbarui A
#         eigenvectors = np.dot(eigenvectors, Q)  # Perbarui eigenvector

#     eigenvalues = [A[i][i] for i in range(n)]  # Eigenvalue diambil dari diagonal
#     return eigenvalues, eigenvectors

def QR(matrix, iterations=10):
    n = len(matrix)
    A = np.array(matrix, dtype=np.float64)  # Gunakan NumPy untuk stabilitas numerik
    eigenvectors = np.eye(n)  # Matriks identitas dengan NumPy

    for _ in range(iterations):
        Q = []
        for i in range(n):
            col = A[:, i]
            for prev in Q:
                scale = np.dot(col, prev)
                col -= scale * prev
            norm = np.linalg.norm(col)
            if norm > 1e-10:  # Hindari pembagian nol
                col = col / norm
            else:
                col = np.zeros_like(col)  # Atur ke nol jika norma terlalu kecil
            Q.append(col)
        Q = np.array(Q).T  # Konversi ke array matriks
        R = np.dot(Q.T, A)
        A = np.dot(R, Q)  # Perbarui A = R * Q
        eigenvectors = np.dot(eigenvectors, Q)

    eigenvalues = np.diag(A)  # Ambil elemen diagonal
    return eigenvalues, eigenvectors.tolist()



def svd(A):
    # Normalisasi matriks
    A = (A - np.mean(A, axis=0)) / (np.std(A, axis=0) + 1e-10)

    # Hitung matriks AAT
    AAT = np.dot(A, transpose(A))

    # Tambahkan regularisasi untuk mencegah singularitas
    AAT += np.eye(A.shape[0]) * 1e-6

    # QR untuk menghitung eigenvalues dan eigenvectors dari AAT
    eigenvalues_AAT, eigenvectors_AAT = QR(AAT)

    # Sort eigenvalues dan eigenvectors
    eigenvalues_AAT, eigenvectors_AAT = sort_eigen(eigenvalues_AAT, eigenvectors_AAT)

    # Matriks U: eigenvectors dari AAT (eigenvec = vektor kolom)
    U = transpose(eigenvectors_AAT)
    return U


