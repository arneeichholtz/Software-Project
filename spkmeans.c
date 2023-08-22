#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
Reading input data, command line goal, and performing goal
Goals: wam -> weighted adjacency L
        ddg -> Diagonal Degree L
        gl -> Graph Laplacian
        jacobi -> input symmetric L, output eigenvectors and -values

Example usage:
compile: gcc spkmeans.c -o spkmeans.out
run: ./spkmeans.out jacobi input_2.txt
*/

double square(double a){
    return a*a;
}

double square_euclid(double * a, double * b, int dim){
    int d;
    double result = 0;
    for(d = 0; d < dim; ++d)
        result = result + square(a[d] - b[d]);
    return result;
}

void set_wam(double ** data, double ** wam, int N, int dim){
    int i, j;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j){
            if(i == j)
                wam[i][j] = 0;
            else
                wam[i][j] = exp(-square_euclid(data[i], data[j], dim)/2); 
        }
}

/* Finds and returns the degree of a data point, ie, the sum of all its weights of connections to other points */
double get_degree(double * weights, int N){
    int n;
    double degree = 0;
    for(n = 0; n < N; ++n)
        degree += weights[n];
    return degree;
}

void set_ddg(double ** wam, double ** ddg, int N){
    int i;
    double degree;
    for(i = 0; i < N; ++i){
        degree = get_degree(wam[i], N);
        ddg[i][i] = degree;
    }
}

void set_gl(double ** wam, double ** ddg, double ** gl, int N){
    int i, j;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            gl[i][j] = ddg[i][j] - wam[i][j];
}

void find_pivot_coords(double ** L, int * coords, int N){
    int n, m;
    double val;
    double max = 0.0;
    int r = 0;
    int c = 0;
    for(n = 0; n < N; ++n)
        for(m = 0; m < N; ++m)
            if(n != m){
                val = fabs(L[n][m]);
                if(val > max){
                    max = val;
                    r = n;
                    c = m;
                }
            }
    coords[0] = r;
    coords[1] = c;
}

int sign(double x){
    if(x < 0)
        return -1;
    else
        return 1;
}

void print_mat(double ** matrix, int N){
    int i, j;
    for(i = 0; i < N; ++i){
        for(j = 0; j < N; ++j){
            if(j != N-1)
                printf("%.4f,", matrix[i][j]);
            else
                printf("%.4f", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_eig_vects(double ** matrix, int N){
    int i, j;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            printf("%f\n", matrix[j][i]);
}

void print_eig_vals(double ** matrix, int N){
    int i;
    for(i = N-1; i >= 0; --i){
        if(i != 0){
            printf("%.4f,", matrix[i][i]);
        }
        else{
            printf("%.4f\n", matrix[i][i]);
        }
    }
}

void set_transpose(double ** L, double ** t, int N){
    int i, j;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            t[j][i] = L[i][j];
}

void mat_mul(double ** a, double ** b, double ** result, int N){
    int i, j, n;
    double res;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j){
            res = 0;
            for(n = 0; n < N; ++n)
                res += a[i][n] * b[n][j];
            result[i][j] = res;
        }
}

int isnotdiagonal(double ** matrix, int N){
    int i, j;
    int sum = 0;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            if(i != j)
                sum += matrix[i][j];
    return sum;
}

/* Returns sum of all squared off-diagonal values */ 
double off(double ** matrix, int N){
    int i, j;
    double sum = 0;
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            if(i != j)
                sum += square(matrix[i][j]);
    return sum;
}

void reset_P(double ** P, int N){
    int i, j;
    for(i = 0; i < N; ++i){
        for(j = 0; j < N; ++j){
            if(i==j){
                P[i][j] = 1;
            }
            else{
                P[i][j] = 0;
            }
        }
    }
}

void copy_mat(double ** copy, double ** original, int N){
    int i, j;
    for(i = 0; i < N; ++i){
        for(j = 0; j < N; ++j){
            copy[i][j] = original[i][j];
        }
    }
}

void jacobi_iter(double ** L, double ** newL, double ** P, double ** trans, double ** interm, int * piv_coords, int N){
    int row = piv_coords[0];
    int col = piv_coords[1];

    double theta = (L[col][col] - L[row][row]) / (2* L[row][col]);
    double t = sign(theta) / (fabs(theta) + sqrt(square(theta) + 1));
    double c = 1 / sqrt(square(t) + 1);
    double s = t * c;

    P[row][col] = s;
    P[col][row] = -s;
    P[row][row] = c;
    P[col][col] = c;

    /*
    Multiplying P^T * L * P
    - P^T is stored in trans 
    - result is stored in interm
    - result is stored in newL
    */
    set_transpose(P, trans, N);  
    mat_mul(trans, L, interm, N);
    mat_mul(interm, P, newL, N);
}

int main(int argc, char **argv)
{
    /* --------------------------
        READING DATA AND ARGS
       -------------------------- */
    
    FILE * fstream;
    char first_line[1000];

    int ch, N, dim, count;
    
    char * goal;
    char * file_name;
    double * data_ptr;
    double ** data;
    double ** wam;
    double ** ddg;
    double ** gl;

    double ** P;
    double ** V;
    double ** newL;
    double ** newV;
    double ** trans;
    double ** interm;
    int * coords;

    double diff, epsilon;
    int rotate, max_rotate;

    int n, d, i; /* Variables for loops */
    
    if(argc != 3){
        printf("An error occurred!");
        exit(1);
    }

    goal = argv[1];
    file_name = argv[2];

    fstream = fopen(file_name, "r");
    N = 0;
    for (ch = getc(fstream); ch != EOF; ch = getc(fstream))
        if (ch == '\n')
            N = N + 1;
    
    count = 0; /* Count of commas in first line to determine dimensions */
    fstream = fopen(file_name, "r");
    fscanf(fstream, "%[^\n]", first_line);
    for(i = 0; first_line[i] != 0; i++)
        count += (first_line[i] == ',');
    dim = count + 1;
    fclose(fstream);

    /* Allocating memory for data -- storing data as contiguous block */
    data_ptr = calloc(N * dim, sizeof(double));
    data = calloc(N, sizeof(double *));
    for(n = 0; n < N; ++n)
        data[n] = data_ptr + n * dim;

    /* Storing data in memory */
    fstream = fopen(file_name, "r");
    for(n = 0; n < N; ++n)
        for(d = 0; d < dim; ++d){
            fscanf(fstream, "%lf", data[n] + d);
            fscanf(fstream, ",");
        }
    fclose(fstream);

    /* ----------------------
        ALGORITHM
       ---------------------- */

    /* weighted adjacency matrix */ 
    wam = calloc(N, sizeof(double *));
    for (i = 0; i < N; ++i)
        wam[i] = calloc(N, sizeof(double));
    set_wam(data, wam, N, dim);
    
    if(!strcmp(goal, "wam")){
        print_mat(wam, N);
        exit(0);
    }

    /* diagonal degree matrix */
    ddg = calloc(N, sizeof(double *));
    for(i = 0; i < N; ++i)
        ddg[i] = calloc(N, sizeof(double));
    set_ddg(wam, ddg, N);

    if(!strcmp(goal, "ddg")){
        print_mat(ddg, N);
        exit(0);
    }

    /* graph laplacian */
    gl = calloc(N, sizeof(double *));
    for(i = 0; i < N; ++i)
        gl[i] = calloc(N, sizeof(double));
    set_gl(wam, ddg, gl, N);

    if(!strcmp(goal, "gl")){
        print_mat(gl, N);
        exit(0);
    }

    if(!strcmp(goal, "jacobi")){
        /* Read symmetric matrix from input file */
        gl = data;

        /* Allocating memory for rotation matrix P */
        P = calloc(N, sizeof(double *));
        for(n = 0; n < N; ++n){
            P[n] = calloc(N, sizeof(double));
        }

        /* Allocating memory for V */
        V = calloc(N, sizeof(double *)); 
        for(n = 0; n < N; ++n){
            V[n] = calloc(N, sizeof(double));
        }

        for(n = 0; n < N; ++n){
            for(i = 0; i < N; ++i){
                if(i == n){
                    P[n][i] = 1; /* Setting diagonal of P to 1 */
                    V[n][i] = 1; /* Setting diagonal of V to 1 */
                }
                else{
                    P[n][i] = 0; 
                    V[n][i] = 0; 
                }
            }
        }

        newL = calloc(N, sizeof(double *)); 
        for(n = 0; n < N; ++n){
            newL[n] = calloc(N, sizeof(double));
        }

        newV = calloc(N, sizeof(double *)); 
        for(n = 0; n < N; ++n){
            newV[n] = calloc(N, sizeof(double));
        }

        trans = calloc(N, sizeof(double *)); 
        for(n = 0; n < N; ++n){
            trans[n] = calloc(N, sizeof(double));
        }

        /* Allocating memory for intermediate matrix prod of P^T * L */
        interm = calloc(N, sizeof(double *));
        for(n = 0; n < N; ++n){
            interm[n] = calloc(N, sizeof(double));
        }

        coords = calloc(2, sizeof(int));

        diff = 1.0;
        epsilon = 0.00001;
        rotate = 0;
        max_rotate = 100;

        while(diff > epsilon && rotate < max_rotate && isnotdiagonal(gl, N)){
            find_pivot_coords(gl, coords, N);
            jacobi_iter(gl, newL, P, trans, interm, coords, N);
            
            diff = off(gl, N) - off(newL, N);
            copy_mat(gl, newL, N);
            mat_mul(V, P, newV, N);

            reset_P(P, N);
            copy_mat(V, newV, N);
            
            rotate += 1;
        }
        print_eig_vals(newL, N);
        set_transpose(V, trans, N);
        print_mat(trans, N);
    }

    else {
        printf("An error occurred!");
        exit(1);
    }

    return 0;

}
