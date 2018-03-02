#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);
extern void square_dgemm_naive (int, double*, double*, double*);
#define MAX_SPEED 42.9  // defining Bridges Max Gflops/s per core with peak TurboBoost Frequency


double wall_time ()
{
#ifdef GETTIMEOFDAY
    struct timeval t;
    gettimeofday (&t, NULL);
    return 1.*t.tv_sec + 1.e-6*t.tv_usec;
#else
    struct timespec t;
    clock_gettime (CLOCK_MONOTONIC, &t);
    return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
#endif
}

void die (const char* message)
{
    perror (message);
    exit (EXIT_FAILURE);
}

void fill (double* p, int n)
{
    for (int i = 0; i < n; ++i)
        p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
}

void absolute_value (double *p, int n)
{
    for (int i = 0; i < n; ++i)
        p[i] = fabs (p[i]);
}

int main() {
    int test_sizes[] = { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
        319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };
    
    int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);

    /* assume last size is also the largest size */
    int nmax = test_sizes[nsizes-1];
    
    double* buf = NULL;
    buf = (double*) malloc (3 * nmax * nmax * sizeof(double));
    if (buf == NULL) die ("failed to allocate largest problem size");
    
    
    /* For each test size */
    double start_time, time, naive_time, delta;
    int iterations = 10;
    for (int isize = 0; isize < nsizes; ++isize)
    {
        int n = test_sizes[isize];
        double* A = buf + 0;
        double* B = A + nmax*nmax;
        double* C = B + nmax*nmax;
        
        fill (A, n*n);
        fill (B, n*n);
        fill (C, n*n);
        for(int i = 0; i < iterations; i++){
            start_time = wall_time();
            square_dgemm_naive(n, A, B, C);
            naive_time = wall_time() - start_time;
        }
        naive_time = naive_time / iterations;
       
        
        for(int i = 0; i < iterations; i++){
            start_time = wall_time();
            square_dgemm (n, A, B, C);
            time = wall_time() - start_time;
        }
        time = time / iterations;
        
        delta = ((time - naive_time) / naive_time) * -100;
        printf ("Blocked: %f, Naive: %f, Delta: %f  Size: %i  \n", time, naive_time , delta, n);
    }
    
    free (buf);
    return 0;
}
