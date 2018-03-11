

/*
 Please include compiler name below (you may also include any other modules you would like to be loaded)

 COMPILER= gnu

 Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

 CC = cc
 OPT = -O3
 CFLAGS = -Wall -std=gnu99 $(OPT)
 MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
 LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

 */
#include <emmintrin.h>
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            double cij = C[i+j*lda];
            for (int k = 0; k < K; ++k)
                cij += A[i+k*lda] * B[k+j*lda];

            C[i+j*lda] = cij;
        }
}

void do_block_fast (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    static double a[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (32)));
//    static double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
    static double temp[4] __attribute__ ((aligned (32))) = {0, 0, 0, 0};

    static __m256d vec1A;
    static __m256d vec1B;
    static __m256d vec1C;
    static __m256d vec2A;
    static __m256d vec2B;
    static __m256d vec2C;
    static __m256d vecCtmp;
    static __m256d vecCtmp2;
//    __m128d vecA1;
//    __m128d vecB1;
//    __m128d vecC1;
//    __m128d vecA2;
//    __m128d vecB2;
//    __m128d vecC2;
//    __m128d vecCtmp;


//
//    make a local aligned copy of A's block
//    for( int j = 0; j < K; j++ )
//        for( int i = 0; i < M; i++ )
//            a[i+j*BLOCK_SIZE] = A[i+j*lda];

    for( int i = 0; i < M; i++ )
        for( int j = 0; j < K; j++ )
            a[j+i*BLOCK_SIZE] = A[i+j*lda];

/* For each row i of A */
    for (int i = 0; i < M; i+=2)
/* For each column j of B */
        for (int j = 0; j < N; j+=2)
        {
/* Compute C(i,j) */
//            double cij = C[i+j*lda];
            double cijA = C[i+j*lda];
            double cijB = C[(i+1)+j*lda];
            double cijC = C[i+(j+1)*lda];
            double cijD = C[(i+1)+(j+1)*lda];


            for (int k = 0; k < K; k+=2){
//                cij += A[i+k*lda] * B[k+j*lda];
//                cij += A[i+1+k+1*lda] * B[k+1+j+1*lda];

                cijA += a[i+k*BLOCK_SIZE] * B[k+j*lda];
                cijA += a[i+(k+1)*BLOCK_SIZE] * B[(k+1)+j*lda];

                cijB += a[i+k*BLOCK_SIZE] * B[k+(j+1)*lda];
                cijB += a[i+(k+1)*BLOCK_SIZE] * B[(k+1)+(j+1)*lda];

                cijC += a[i+1+k*BLOCK_SIZE] * B[k+(j+1)*lda];
                cijC += a[i+(k+1)*BLOCK_SIZE] * B[(k+1)+(j+1)*lda];

                cijD += a[(i+1)+k*BLOCK_SIZE] * B[k+(j+1)*lda];
                cijD += a[(i+1)+(k+1)*BLOCK_SIZE] * B[(k+1)+(j+1)*lda];

//                vec1A = _mm256_load_pd (&a[k+i*BLOCK_SIZE]);
//                vec1B = _mm256_loadu_pd (&B[k+j*lda]);
//                vec2A = _mm256_load_pd (&a[k+4+i*BLOCK_SIZE]);
//                vec2B = _mm256_loadu_pd (&B[k+4+j*lda]);
//                vec1C = _mm256_mul_pd(vec1A, vec1B);
//                vec2C = _mm256_mul_pd(vec2A, vec2B);
//                vecCtmp = _mm256_add_pd(vec1C,vec2C);
//                _mm256_store_pd(&temp[0], vecCtmp);
//                cij += temp[0];
//                cij += temp[1];
//                cij += temp[2];
//                cij += temp[3];

            }
            C[i+j*lda] = cijA;
            C[(i+1)+j*lda] = cijB;
            C[i+(j+1)*lda] = cijC;
            C[(i+1)+(j+1)*lda] = cijD;
//            C[i+j*lda] = cij;
        }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* C)
{
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE, lda-i);
                int N = min (BLOCK_SIZE, lda-j);
                int K = min (BLOCK_SIZE, lda-k);

                /* Perform individual block dgemm */
                if((M % BLOCK_SIZE == 0) && (N % BLOCK_SIZE == 0) && (K % BLOCK_SIZE == 0))
                {
                    do_block_fast(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
                }else{
                    do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
                }
            }
}

