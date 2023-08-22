# Author: Arne Eichholtz
# Example usage
# setup:    python3 setup.py build_ext --inplace
# running:  python3 spkmeans.py spk input_1.txt OR python3 spkmeans.py 2 jacobi input_3.txt
# possible goals: wam, ddg, gl, jacobi, spk


import numpy as np
import pandas as pd
import sys
import mykmeanssp

# Weighted adjacency matrix
def wam(data, N):
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                W[i, j] = 0
            else:
                W[i, j] = np.exp(-square_euclid(data[i], data[j])/2)
    return W

# Diagonal degree matrix
def ddg(W, N):
    D = np.zeros((N, N))
    np.fill_diagonal(D, [np.sum(W[i]) for i in range(N)])
    return D

# Graph Laplacian
def gl(D, W):
    return D - W

def square_euclid(a, b):
    return np.sum(np.square(a - b))

# Boolean to check if matrix L is diagonal matrix, i.e., all non-diagonal elements are 0.
def isdiagonal(L):
    L_copy = L.copy()
    np.fill_diagonal(L_copy, 0)
    if np.sum(L_copy) == 0:
        return True
    else:
        return False

# Returns the sum of all squared off-diagonal values
def off(A):
    A_copy = A.copy()
    np.fill_diagonal(A_copy, 0)
    return np.sum(A_copy**2)

# Returns -1 if x < 0 and 1 if x >= 0
def sign(x):
    if x < 0:
        return -1
    else:
        return 1

# Make all zeros positive in matrix L, so change -0 to 0
def positive_zero(L):
    return np.where(L == -0, 0, L)

def get_eigen_vals(L):
    return [L[i, i] for i in range(L.shape[0]-1, -1, -1)]

def get_eigen_vects(V):
    return V.tolist()

# Returns list V, which is a flattened N x N matrix, in formatted string 
def matrix_as_str(V, N):
    str = ""
    for i in range(0, N):
        for j in range(0, N):
            if(j != N-1):
                str += f"{V[(i*N)+j]:.4f},"
            else:
                str += f"{V[(i*N)+j]:.4f}"
        if(i != N-1):
            str+="\n"
    return str
            
# Formats vals in 1-D list V into string with values separated by commas, beginning at the end of V
def vals_as_str(V):
    str = ""
    for i in range(len(V) - 1, -1, -1):
        if i != 0:
            str += f"{V[i]:.4f},"
        else:
           str += f"{V[i]:.4f}"
    return str

# Returns np array V in string format, separated by commas
def vects_str(V):
    str = ""
    for r in range(V.shape[0]):
        for c in range(V.shape[1]):
            if c != V.shape[1] -1:
                str += f"{V[r, c]:.4f},"
            else:
                str += f"{V[r, c]:.4f}"
        if r != V.shape[0] - 1:
            str += "\n"
    return str

# Convert 1d np array of indices to str
def inds_to_str(inds):
    str = ""
    for i in range(inds.shape[0] - 1):
        str = str+f"{inds[i]:.0f},"
    return str+f"{inds[-1]:.0f}"

# Convert 1d-list of centroids with length K * K to formatted string
def centroidslst_to_str(centr, K):
    str = ""
    for i, cent in enumerate(centr):
        if i != K - 1:
            str += f"{cent:.4f},"
        else:
            str+= f"{cent:.4f}"
    return str

# Find pivot (largest absolute off-diagonal value) coordinates of matrix L with dim(L) = N (N is nr of datapoints)
def pivot_coords(L, N):
    L_copy = np.abs(L.copy()) # Copy absolute values
    np.fill_diagonal(L_copy, -np.inf) # Set diagonal to -inf so that the values are not picked by argmax() 
    max_ind = np.argmax(L_copy) # max_ind is index in flattened array

    # Pivot row and column
    row = int(max_ind / N)
    col = max_ind % N
    return row, col

# Flatten nd-list into 1d-list
def flatten(lst):
    return list(np.concatenate(lst).flat)

# Single iteration of jacobi algorithm  
def jacobi_iter(L, N):
    row, col = pivot_coords(L, N)

    # Variables for rotation matrix
    theta = (L[col, col] - L[row, row]) / (2 * L[row, col])
    t = sign(theta) / (np.abs(theta) + np.sqrt(theta**2 + 1))
    c = 1 / np.sqrt(t**2 + 1)
    s = t * c

    # Rotation matrix
    P = np.eye(N)
    P[row, col] = s # s on (i, j)
    P[col, row] = -s # -s on (i, j)^T = (j, i)
    P[row, row] = c
    P[col, col] = c

    new_L = np.transpose(P) @ L @ P
    return new_L, P

def euclid_dist(row, centroid):
    return np.linalg.norm(row - centroid)

# Takes np array of sorted eigen values and finds k, the index of the largest difference between consecutive eigenvalues
def eigen_gap_k(sorted_vals):
  delta = 0
  K = 0
  for i in range(0, sorted_vals.shape[0] - 1):
    new_delta = np.abs(sorted_vals[i] - sorted_vals[i+1])
    if new_delta > delta:
      delta = new_delta
      K = i
  return K

# Takes np arrays eigenvalues (vals) and eigenvectors (vects) and projects data onto first K eigenvectors, 
# where K is the index of the largest difference between sorted eigenvalues
def first_k_vects(vals, vects):
  inds = np.argsort(vals)
  vects = vects[:, inds]
  sorted_vals = np.sort(vals)
  K = eigen_gap_k(sorted_vals)
  return K, vects[:, 0:K+1]

# Projects data onto first k eigenvectors, where K is given -- K not incremented by 1 because it is not index here
def first_k_vects_given(vals, vects, K):
  inds = np.argsort(vals)
  vects = vects[:, inds]
  return vects[:, 0:K]

# Initializing k centroids for k-means++
def init_centroids(K, df):
    indices = np.zeros(K)

    # Initializing first centroid
    index = np.random.randint(low=0, high=df.shape[0])
    indices[0] = index
    centroids = df.iloc[[index]] # DataFrame to save the centroids

    # Initializing other K-1 centroids
    for k in range(1, K):
        # distances, D(x), holds the distance for each point to its closes centroid
        distances = df.apply(lambda row: min([euclid_dist(row, centroids.iloc[i]) for i in range(centroids.shape[0])]), axis=1)
        sum = np.sum(distances)
        distribution = distances.apply(lambda row: row / sum)

        new_index = np.random.choice(a=df.shape[0], p=distribution) # df.shape[0] = N, so a=np.arange(N)
        new_centr = df.iloc[[new_index]] # double brackets means return type is df, not series
        indices[k] = new_index

        # Update centroids
        centroids = pd.concat([centroids, new_centr], ignore_index=True)

    return indices, centroids

if __name__ == "__main__":

    np.random.seed(0)

    # READING USER COMMAND ARGUMENTS
    if len(sys.argv) == 3: # Default is that k is not given, so use eigengap heuristic
        try:
            k = 1
            goal = sys.argv[1]
            file_name = sys.argv[2]
        except:
            print("An error occurred!")
            exit()
    elif len(sys.argv) == 4: # k is given
        try:
            k = int(sys.argv[1])
            goal = sys.argv[2]
            file_name = sys.argv[3]
        except:
            print("An error occurred!")
            exit
    else:
        print("An error occurred!")
        exit()

    data = np.loadtxt(file_name, delimiter=",")
    N = data.shape[0]
    dim = data.shape[1]
    data_flat = flatten(data.tolist())
    
    # Eigengap heuristic
    gl = mykmeanssp.gl(data_flat, N, dim)
    eig = mykmeanssp.jacobi(gl, N)
    vals = eig[0]
    vals_arr = np.array(vals)
    vects = eig[1]
    vects_arr = np.transpose(np.array(vects).reshape((N, N))) # each row in vects_arr to be treated as data point and clustered using kmeans

    if len(sys.argv) == 3:
        k, eigen_data = first_k_vects(vals_arr, vects_arr) # Data projected onto first k eigenvectors
        k = k + 1 # k is the index of the largest difference, for number of clusters it must be incremented by 1.
    elif len(sys.argv) == 4:
        eigen_data = first_k_vects_given(vals_arr, vects_arr, k)
    
    if k > N:
        print("An error occurred!")
        exit()

    if goal == "wam":
        wam = mykmeanssp.wam(data_flat, N, dim)
        print(matrix_as_str(wam, N))
        exit()
    
    elif goal == "ddg":
        ddg = mykmeanssp.ddg(data_flat, N, dim)
        print(matrix_as_str(ddg, N))
        exit()
    
    elif goal == "gl":
        gl = mykmeanssp.gl(data_flat, N, dim)
        print(matrix_as_str(gl, N))
        exit()

    elif goal == "jacobi":
        eig = mykmeanssp.jacobi(data_flat, N) # eig[0] contains list with eigenvalues, eig[1] contains flattened list with eigenvectors
        vals = eig[0]
        vects = eig[1]
        vects_arr = np.transpose(np.array(vects).reshape((N, N))) # reshape np array and transpose so eigenvectors are columns in result
        print(vals_as_str(vals))
        print(vects_str(vects_arr))
        exit()

    elif goal == "spk":

        # For the eigengap heuristic, the code below is moved to above the if goal == "wam" statement, but like this it is actually a bit weird, because for 
        # the other goals you don't even need k. You can have a look

        # gl = mykmeanssp.gl(data_flat, N, dim)
        # eig = mykmeanssp.jacobi(gl, N)

        # vals = eig[0]
        # vals_arr = np.array(vals)
        # vects = eig[1]
        # vects_arr = np.transpose(np.array(vects).reshape((N, N))) # each row in vects_arr to be treated as data point and clustered using kmeans
        
        # if len(sys.argv) == 3:
        #     K, eigen_data = first_k_vects(vals_arr, vects_arr) # Data projected onto first k eigenvectors
        #     K = K + 1 # K is the index of the largest difference, for number of clusters it must be incremented by 1.
        # elif len(sys.argv) == 4:
        #     K = int(sys.argv[1])
        #     eigen_data = first_k_vects_given(vals_arr, vects_arr, K)
        # else:
        #     print("An error occurred!")
        #     exit()

        eigen_data_df = pd.DataFrame(eigen_data) # df is required for finding initial centroids
        indices, init_cent = init_centroids(k, eigen_data_df)
        print(inds_to_str(indices))
        
        # Flatten centroids and eigenvectors to pass to mykmeanssp function
        cent_flat = flatten(init_cent.values.tolist()) 
        data_flat = flatten(eigen_data)
        
        centroids = mykmeanssp.spk(data_flat, cent_flat, k, N)
        print(matrix_as_str(centroids, k))
        exit()

    else: # Invalid command for goal
        print("An error occurred!") 
        exit()

