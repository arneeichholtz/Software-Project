#ifndef SPKMEANS_H
#define SPKMEANS_H

double square(double a);

double square_euclid(double * a, double * b, int dim);

void set_wam(double ** data, double ** wam, int N, int dim);

double get_degree(double * weights, int N);

void set_ddg(double ** wam, double ** ddg, int N);

void set_gl(double ** wam, double ** ddg, double ** gl, int N);

void find_pivot_coords(double ** L, int * coords, int N);

int sign(double x);

void print_mat(double ** matrix, int N);

void print_eig_vects(double ** matrix, int N);

void print_eig_vals(double ** matrix, int N);

void set_transpose(double ** L, double ** t, int N);

void mat_mul(double ** a, double ** b, double ** result, int N);

int isnotdiagonal(double ** matrix, int N);

double off(double ** matrix, int N);

void jacobi_iter(double ** L, double ** newL, double ** P, double ** trans, double ** interm, int * piv_coords, int N);

void copy_mat(double ** copy, double ** original, int N);

void reset_P(double ** P, int N);

#endif