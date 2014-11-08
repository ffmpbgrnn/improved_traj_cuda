// This is the original license from GSL for this file

/* blas/blas.c
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2001, 2009 Gerard Jungman & Brian
 * Gough
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* GSL implementation of BLAS operations for vectors and dense
 * matrices.  Note that GSL native storage is row-major.  */

 // This is basically a copy and paste of GSL SVD function, with some modifications
 // to get it working on CUDA. Mostly function name changes and re-implementation of external functions and classes


// Using GSL_FLT_EPSILON, find and replace with the float version if your card supports it
#define GPU_EPSILON        1.1920928955078125e-07 // float
//#define EPSILON        2.2204460492503131e-16 // double

// Change as required
#define GPU_MAX_ROWS 9
#define GPU_MAX_COLS 9

struct GPU_Matrix
{
    float data[GPU_MAX_ROWS*GPU_MAX_COLS];
    int rows;
    int cols;
};

struct GPU_Vector
{
    float data[GPU_MAX_ROWS];
    int size;
};

// Call this function in the kernel
// This computes only Q,S, where Q (aka V) is the right side matrix and S the singular values, A is modified and used as temp working space
__device__ bool linalg_SV_decomp_jacobi (GPU_Matrix *A, GPU_Matrix *Q, GPU_Vector *S);

__device__ inline void matrix_set_identity(GPU_Matrix *A);
__device__ inline void matrix_column(GPU_Matrix *A, int col, GPU_Vector *V);
__device__ inline void matrix_set(GPU_Matrix *A, int row, int col, float val);
__device__ inline float matrix_get(GPU_Matrix *A, int row, int col);
__device__ inline void vector_set(GPU_Vector *V, int i, float v);
__device__ inline float vector_get(GPU_Vector *V, int i);
__device__ inline void vector_set_zero(GPU_Vector *V);
__device__ inline void vector_scale(GPU_Vector *V, float scale);
__device__ inline void ddot(const GPU_Vector *a, const GPU_Vector *b, float *result);
__device__ inline float hypot2 (const float x, const float y); // hypot already defined in cuda/include/math_functions.h
__device__ inline float dnrm2 (GPU_Vector *v);

__device__ void matrix_set_identity(GPU_Matrix *A)
{
    float *data = A->data;

    for(int i=0; i < A->rows; i++) {
        for(int j=0; j < A->cols; j++) {
            if(i == j) {
                *data = 1.0;
            }
            else {
                *data = 0.0;
            }

            data++;
        }
    }
}

__device__ void matrix_column(GPU_Matrix *A, int col, GPU_Vector *V)
{
    for(int y=0; y < A->rows; y++) {
        V->data[y] = A->data[y*A->cols + col];
    }

    V->size = A->rows;
}

__device__ void matrix_set(GPU_Matrix *A, int row, int col, float val)
{
    A->data[row*A->cols + col] = val;
}

__device__ float matrix_get(GPU_Matrix *A, int row, int col)
{
    return A->data[row*A->cols + col];
}

__device__ void vector_set(GPU_Vector *V, int i, float v)
{
    V->data[i] = v;
}

__device__ float vector_get(GPU_Vector *V, int i)
{
    return V->data[i];
}

__device__ void vector_set_zero(GPU_Vector *V)
{
    for(int i=0; i < V->size; i++) {
        V->data[i] = 0.0;
    }
}

__device__ void vector_scale(GPU_Vector *V, float scale)
{
    for(int i=0; i < V->size; i++) {
        V->data[i] *= scale;
    }
}

__device__ void ddot(const GPU_Vector *a, const GPU_Vector *b, float *result)
{
    *result = 0;

    for(int i=0; i < a->size; i++) {
        *result += a->data[i]*b->data[i];
    }
}

__device__ float hypot2 (const float x, const float y)
{
    float xabs = fabs(x) ;
    float yabs = fabs(y) ;
    float min, max;

    if (xabs < yabs) {
        min = xabs ;
        max = yabs ;
    } else {
        min = yabs ;
        max = xabs ;
    }

    if (min == 0)
    {
      return max ;
    }

    {
    float u = min / max ;
    return max * sqrt (1 + u * u) ;
    }
}

__device__ float dnrm2 (GPU_Vector *v)
{
    const int N = v->size;
    const float *X = v->data;
    const int incX = 1;

    float scale = 0.0;
    float ssq = 1.0;
    int i;
    int ix = 0;

  if (N <= 0 || incX <= 0) {
    return 0;
  } else if (N == 1) {
    return fabs(X[0]);
  }

  for (i = 0; i < N; i++) {
    const float x = X[ix];

    if (x != 0.0) {
      const float ax = fabs(x);

      if (scale < ax) {
        ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
        scale = ax;
      } else {
        ssq += (ax / scale) * (ax / scale);
      }
    }

    ix += incX;
  }

  return scale * sqrt(ssq);
}

__device__ bool linalg_SV_decomp_jacobi (GPU_Matrix *A, GPU_Matrix *Q, GPU_Vector *S)
{
      const size_t M = A->rows;
      const size_t N = A->cols;
      size_t i, j, k;
        GPU_Vector cj;
        GPU_Vector ck;
        GPU_Vector column;

      /* Initialize the rotation counter and the sweep counter. */
      int count = 1;
      int sweep = 0;
      int sweepmax = 5*N;

      float tolerance = 10 * M * GPU_EPSILON;

      /* Always do at least 12 sweeps. */
      sweepmax = max (sweepmax, 12);

      /* Set Q to the identity matrix. */
      matrix_set_identity (Q);

      /* Store the column error estimates in S, for use during the
         orthogonalization */

      for (j = 0; j < N; j++)
        {

          matrix_column(A, j, &cj);

          float sj = dnrm2 (&cj);

          S->data[j] = GPU_EPSILON * sj;
        }

      /* Orthogonalize A by plane rotations. */

      while (count > 0 && sweep <= sweepmax)
        {
          /* Initialize rotation counter. */
          count = N * (N - 1) / 2;

          for (j = 0; j < N - 1; j++)
            {
              for (k = j + 1; k < N; k++)
                {
                  float a = 0.0;
                  float b = 0.0;
                  float p = 0.0;
                  float q = 0.0;
                  float cosine, sine;
                  float v;
                  float abserr_a, abserr_b;
                  int sorted, orthog, noisya, noisyb;

                  //gsl_vector_view cj = gsl_matrix_column (A, j);
                  matrix_column(A, j, &cj);

                  //gsl_vector_view ck = gsl_matrix_column (A, k);
                    matrix_column(A, k, &ck);

                  ddot (&cj, &ck, &p);
                  p *= 2.0 ;  /* equation 9a:  p = 2 x.y */

                  a = dnrm2 (&cj);
                  b = dnrm2 (&ck);

                  q = a * a - b * b;
                  v = hypot2(p, q);

                  /* test for columns j,k orthogonal, or dominant errors */

                  abserr_a = vector_get(S,j);
                  abserr_b = vector_get(S,k);

                  sorted = a >= b;
                  orthog = (fabs (p) <= tolerance * (a * b));
                  noisya = (a < abserr_a);
                  noisyb = (b < abserr_b);

                  if (sorted && (orthog || noisya || noisyb))
                    {
                      count--;
                      continue;
                    }

                  /* calculate rotation angles */
                  if (v == 0 || !sorted)
                    {
                      cosine = 0.0;
                      sine = 1.0;
                    }
                  else
                    {
                      cosine = sqrt((v + q) / (2.0 * v));
                      sine = p / (2.0 * v * cosine);
                    }

                  /* apply rotation to A */
                  for (i = 0; i < M; i++)
                    {
                      const float Aik = matrix_get (A, i, k);
                      const float Aij = matrix_get (A, i, j);
                      matrix_set (A, i, j, Aij * cosine + Aik * sine);
                      matrix_set (A, i, k, -Aij * sine + Aik * cosine);
                    }

                  vector_set(S, j, fabs(cosine) * abserr_a + fabs(sine) * abserr_b);
                  vector_set(S, k, fabs(sine) * abserr_a + fabs(cosine) * abserr_b);

                  /* apply rotation to Q */
                  for (i = 0; i < N; i++)
                    {
                      const float Qij = matrix_get (Q, i, j);
                      const float Qik = matrix_get (Q, i, k);
                      matrix_set (Q, i, j, Qij * cosine + Qik * sine);
                      matrix_set (Q, i, k, -Qij * sine + Qik * cosine);
                    }
                }
            }

          /* Sweep completed. */
          sweep++;
        }

      /*
       * Orthogonalization complete. Compute singular values.
       */

      {
        float prev_norm = -1.0;

        for (j = 0; j < N; j++)
          {
            matrix_column (A, j, &column);
            float norm = dnrm2 (&column);

            /* Determine if singular value is zero, according to the
               criteria used in the main loop above (i.e. comparison
               with norm of previous column). */

            if (norm == 0.0 || prev_norm == 0.0
                || (j > 0 && norm <= tolerance * prev_norm))
              {
                vector_set (S, j, 0.0);     /* singular */
                vector_set_zero (&column);   /* annihilate column */

                prev_norm = 0.0;
              }
            else
              {
                vector_set (S, j, norm);    /* non-singular */
                vector_scale (&column, 1.0 / norm);  /* normalize column */

                prev_norm = norm;
              }
          }
      }

      if (count > 0)
        {
          return false;
        }

      return true;
}
