#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

/* Calculates the euclideon distance between coordinate (coord) and centroid (centr), returns a scalar */
double euclid_dist(double * coord, double * centr, int dim){
    int d;
    double dist;
    dist = 0.0;
    for(d = 0; d < dim; ++d){
        dist = dist + square(*(coord + d) - *(centr + d));
    }
    return sqrt(dist);
}

/* Finds minimum value out of K values */
double min_distance(double * distances, int K){
    int m;
    double min_dist;
    min_dist = *(distances);
    for(m = 1; m < K; ++m){
        double dist = *(distances + m);
        if(dist < min_dist){
            min_dist = dist;
        }
    }
    return min_dist;
}

/* Finds index of closest centroid, i.e., index of smallest value in distances */
int min_centr_ind(double * distances, int K){
    int min_ind;
    double min_dist;
    min_dist = min_distance(distances, K);
    for(min_ind = 0; min_ind < K; ++min_ind){
        if(min_dist == *(distances + min_ind)){
            return min_ind;
        }
    }
    return -1;
}

/* Finds maximum value out of K values */
double find_max_diff(double * diffs, int K){
    int k;
    double max_diff;
    max_diff = *diffs;
    for(k = 1; k < K; ++k){
        if (max_diff < *(diffs + k)){
            max_diff = *(diffs + k);
        }
    }
    return max_diff;
}

/* K-means implementation in C */ 
double ** fit(double ** data, double ** centroids, int K, int max_iter, double epsilon, int N, int dim){
    int curr_iter = 0;
    double max_diff = 1;
    int n, d, k;

    double *** cluster_values;
    double * coord;
    double * distances;
    double * centr;
    double dist;
    int min_centr;

    double ** new_centroids;
    double diff;
    double * diffs;
    int non_zero;
    double * dim_means;

    while (curr_iter < max_iter && epsilon < max_diff){
        cluster_values = calloc(K, sizeof(double **)); // Initialize cluster values memory
        for(k = 0; k < K; ++k){
            cluster_values[k] = calloc(N, sizeof(double *)); // For each k, block of size N to store possible coordinates
        }

        // Looping over data points 
        for(n = 0; n < N; ++n){
            coord = data[n];
            distances = calloc(K, sizeof(double)); // Distances for data point to each cluster

            // Looping over centroids
            for(k = 0; k < K; ++k){
                centr = centroids[k];
                dist = euclid_dist(coord, centr, dim);
                distances[k] = dist;
            }
            min_centr = min_centr_ind(distances, K); // Index of closest centroid for data point n
            cluster_values[min_centr][n] = coord;
            free(distances);
        }

        // Calculate average of centroids and update them 
        new_centroids = calloc(K, sizeof(double *));
        diffs = calloc(K, sizeof(double)); // Difference between updated centroid and old centroid

        // Looping over clusters to find average
        for (k = 0; k < K; ++k){
            non_zero = 0;
            dim_means = calloc(dim, sizeof(double)); // Mean per dimension of the points of a cluster
            // Looping over data points within cluster k 
            for (n = 0; n < N; ++n){
                coord = cluster_values[k][n];
                if (coord != NULL){
                    ++non_zero;
                    for(d = 0; d < dim; ++d){
                        dim_means[d] += *(coord + d);
                    }
                }   
            }

            for(d = 0; d < dim; ++d){
                dim_means[d] = dim_means[d] / non_zero;
            }

            diff = euclid_dist(dim_means, centroids[k], dim);
            diffs[k] = diff;
            new_centroids[k] = dim_means;
        }
        max_diff = find_max_diff(diffs, K);
        centroids = new_centroids;
        curr_iter = curr_iter + 1;
    }
    return centroids;
}

static PyObject * wam_build(PyObject *self, PyObject *args){
    PyObject *data_lst; /* PyObject pointer (unparsed) to store data */
    PyObject *point; /* data point (1, d) */
    double point_dim; /* point dim value (1) */
    int N, dim;
    int n, d, m; /* Variables for loops */
    double ** data;
    double ** wam;
    PyObject *py_val; /* PyObject pointer to store value of an entry in the wam */
    PyObject *py_wam;
    double val;

    if(!PyArg_ParseTuple(args, "Oii", &data_lst, &N, &dim))
        return NULL;
    
    data = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        data[n] = calloc(dim, sizeof(double)); 

    /* Storing datapoints from PyObject data_lst in memory */
    for(n = 0; n < N; ++n){
        for(d = 0; d < dim; ++d){
            point = PyList_GetItem(data_lst, (n*dim)+d);
            point_dim = PyFloat_AsDouble(point);
            data[n][d] = point_dim;
        }
    }

    wam = calloc(N, sizeof(double *));
    for (n = 0; n < N; ++n)
        wam[n] = calloc(N, sizeof(double));
    set_wam(data, wam, N, dim);

    /* Storing wam ** into a list that can be returned to Python */
    py_wam = PyList_New(N*N);
    
    for (n = 0; n < N; ++n){
        for(m = 0; m < N; ++m){
            val = wam[n][m];
            py_val = Py_BuildValue("d", val);
            PyList_SetItem(py_wam, (n*N)+m, py_val);
        }
    }
    return py_wam;
}

static PyObject * ddg_build(PyObject *self, PyObject *args){
    PyObject *data_lst;
    PyObject *point; 
    double point_dim; 
    int N, dim;
    int n, d, m;
    double ** data;
    double ** wam;
    double ** ddg;
    PyObject *py_val;
    PyObject *py_ddg;
    double val;

    if(!PyArg_ParseTuple(args, "Oii", &data_lst, &N, &dim))
        return NULL;
    
    data = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        data[n] = calloc(dim, sizeof(double)); 

    for(n = 0; n < N; ++n){
        for(d = 0; d < dim; ++d){
            point = PyList_GetItem(data_lst, (n*dim)+d);
            point_dim = PyFloat_AsDouble(point);
            data[n][d] = point_dim;
        }
    }

    /* Allocating memory and setting weighted adjacency matrix -- this is required for ddg */
    wam = calloc(N, sizeof(double *));
    for (n = 0; n < N; ++n)
        wam[n] = calloc(N, sizeof(double));
    set_wam(data, wam, N, dim);
    
    ddg = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        ddg[n] = calloc(N, sizeof(double));
    set_ddg(wam, ddg, N);

    py_ddg = PyList_New(N*N);
    
    for (n = 0; n < N; ++n){
        for(m = 0; m < N; ++m){
            val = ddg[n][m];
            py_val = Py_BuildValue("d", val);
            PyList_SetItem(py_ddg, (n*N)+m, py_val);
        }
    }
    return py_ddg;
}

static PyObject * gl_build(PyObject *self, PyObject *args) { 
    PyObject * data_lst; 
    PyObject * point; 
    double point_dim;
    int N, dim;
    int n, d, m;
    double ** data;
    double ** wam;
    double ** ddg;
    double ** gl;
    PyObject *py_val;
    PyObject *py_gl;
    double val;

    if(!PyArg_ParseTuple(args, "Oii", &data_lst, &N, &dim))
        return NULL;

    data = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        data[n] = calloc(dim, sizeof(double)); 

    for(n = 0; n < N; ++n){
        for(d = 0; d < dim; ++d){
            point = PyList_GetItem(data_lst, (n*dim)+d);
            point_dim = PyFloat_AsDouble(point);
            data[n][d] = point_dim;
        }
    }

    /* Allocating memory and setting weighted adjacency matrix -- this is required for gl */
    wam = calloc(N, sizeof(double *));
    for (n = 0; n < N; ++n)
        wam[n] = calloc(N, sizeof(double));
    set_wam(data, wam, N, dim);

    /* Allocating memory and setting diagonal degree matrix -- this is required for gl */
    ddg = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        ddg[n] = calloc(N, sizeof(double));
    set_ddg(wam, ddg, N);

    gl = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        gl[n] = calloc(N, sizeof(double));
    set_gl(wam, ddg, gl, N);

    py_gl = PyList_New(N*N);
    
    for (n = 0; n < N; ++n){
        for(m = 0; m < N; ++m){
            val = gl[n][m];
            py_val = Py_BuildValue("d", val);
            PyList_SetItem(py_gl, (n*N)+m, py_val);
        }
    }
    return py_gl;
}

static PyObject * jacobi_build(PyObject *self, PyObject *args) {
    PyObject * matrix_lst;
    PyObject * row;
    double row_val;
    int N;
    int n, m;
    double ** matrix;
    double ** P;
    double ** V;
    double ** newL;
    double ** newV;
    double ** trans;
    double ** interm;

    int * coords;
    double diff, epsilon;
    int rotate, max_rotate;

    PyObject * py_val;
    PyObject * py_eig;
    PyObject * py_eigenvals;
    PyObject * py_eigenvects;
    double eigen_val;
    double eig_vect_val;

    if(!PyArg_ParseTuple(args, "Oi", &matrix_lst, &N))
        return NULL;

    matrix = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        matrix[n] = calloc(N, sizeof(double));

    for(n = 0; n < N; ++n){
        for(m = 0; m < N; ++m){
            row = PyList_GetItem(matrix_lst, (n*N)+m);
            row_val = PyFloat_AsDouble(row);
            matrix[n][m] = row_val;
        }
    }

    P = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        P[n] = calloc(N, sizeof(double));

    V = calloc(N, sizeof(double *)); 
    for(n = 0; n < N; ++n)
        V[n] = calloc(N, sizeof(double));

    for(n = 0; n < N; ++n){
        P[n][n] = 1;
        V[n][n] = 1;
    }

    newL = calloc(N, sizeof(double *)); 
    for(n = 0; n < N; ++n)
        newL[n] = calloc(N, sizeof(double));

    newV = calloc(N, sizeof(double *)); 
    for(n = 0; n < N; ++n)
        newV[n] = calloc(N, sizeof(double));
    
    trans = calloc(N, sizeof(double *)); 
    for(n = 0; n < N; ++n)
        trans[n] = calloc(N, sizeof(double));

    interm = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        interm[n] = calloc(N, sizeof(double));

    coords = calloc(2, sizeof(int));
    
    diff = 1.0;
    epsilon = 0.00001;
    rotate = 0;
    max_rotate = 100;

    while(diff > epsilon && rotate < max_rotate && isnotdiagonal(matrix, N)){
        find_pivot_coords(matrix, coords, N);
        jacobi_iter(matrix, newL, P, trans, interm, coords, N);
        
        diff = off(matrix, N) - off(newL, N);
        copy_mat(matrix, newL, N);
        mat_mul(V, P, newV, N);
        reset_P(P, N);
        copy_mat(V, newV, N);
    
        rotate += 1;
    }

    /* Storing newL (eigenvalues) and V (eigenvectors) */
    py_eig = PyList_New(2);
    py_eigenvals = PyList_New(N);
    py_eigenvects = PyList_New(N*N);
    
    for (n = 0; n < N; ++n){
        for(m = 0; m < N; ++m){
            if(n == m){
                eigen_val = newL[n][m];
                py_val = Py_BuildValue("d", eigen_val);
                PyList_SetItem(py_eigenvals, n, py_val);
            }
            eig_vect_val = V[n][m];
            py_val = Py_BuildValue("d", eig_vect_val);
            PyList_SetItem(py_eigenvects, (n*N)+m, py_val);
        }
    }

    PyList_SetItem(py_eig, 0, py_eigenvals);
    PyList_SetItem(py_eig, 1, py_eigenvects);
    return py_eig;
}

static PyObject * spk_build(PyObject *self, PyObject *args){
    PyObject *data_lst;
    PyObject *cent_lst;
    PyObject *py_value;
    double c_value;
    int K, N;
    int n, k, l;
    double ** data;
    double ** init_centroids;
    int max_iter;
    double epsilon;
    int dim;
    double ** centroids;
    PyObject *py_cent;

    if(!PyArg_ParseTuple(args, "OOii", &data_lst, &cent_lst, &K, &N)) 
        return NULL;

    /* Allocating memory for data and storing in memory */
    data = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        data[n] = calloc(K, sizeof(double)); /* data is shaped N x K */
    
    for(n = 0; n < N; ++n){
        for(k = 0; k < K; ++k){
            py_value = PyList_GetItem(data_lst, (n*K)+k);
            c_value = PyFloat_AsDouble(py_value);
            data[n][k] = c_value;
        }
    }

    /* Allocating memory for initial centroids and storing in memory */
    init_centroids = calloc(K, sizeof(double *));
    for(k = 0; k < K; ++k)
        init_centroids[k] = calloc(K, sizeof(double)); /* init_centroids is shaped K x K */
    
    for(k = 0; k < K; ++k){
        for(l = 0; l < K; ++l){
            py_value = PyList_GetItem(cent_lst, (k*K)+l);
            c_value = PyFloat_AsDouble(py_value);
            init_centroids[k][l] = c_value;
        }
    }

    max_iter = 300;
    epsilon = 0.0;
    dim = K; /* data is shaped N x K, so K is the dimensionality */
    
    centroids = fit(data, init_centroids, K, max_iter, epsilon, N, dim);

    py_cent = PyList_New(K*K); /* PyObject pointer that acts as a list to store final centroids to return to Python code */

    for(k = 0; k < K; ++k){
        for(l = 0; l < K; ++l){
            c_value = centroids[k][l];
            py_value = Py_BuildValue("d", c_value);
            PyList_SetItem(py_cent, (k*K)+l, py_value);
        }
    }
    return py_cent;
}

static PyMethodDef spkmeansMethods[] = {
    {"wam", /* Python name that will be used */ 
     (PyCFunction) wam_build, /* The C function that takes the Python arguments and builds the C function */
     METH_VARARGS,           
     PyDoc_STR("Find weighted adjacency matrix for given data")},

    {"ddg", (PyCFunction) ddg_build, METH_VARARGS, PyDoc_STR("Find diagonal degree matrix for given data")}, 
    {"gl", (PyCFunction) gl_build, METH_VARARGS, PyDoc_STR("Find graph laplacian for given data")}, 
    {"jacobi", (PyCFunction) jacobi_build, METH_VARARGS, PyDoc_STR("Find eigen vals and vects for symmetrix matrix")},
    {"spk", (PyCFunction) spk_build, METH_VARARGS, PyDoc_STR("Do spectral K-means clustering")},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef kmeansspmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* Name of module */
    "This module is about testing the spectral k means module", /* Module documentation */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    spkmeansMethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&kmeansspmodule);
    if (!m) {
        return NULL;
    }
    return m;
}