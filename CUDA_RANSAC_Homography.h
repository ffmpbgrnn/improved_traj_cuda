/*
Copyright 2011 Nghia Ho. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY NGHIA HO ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BY NGHIA HO OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Nghia Ho.
*/

#ifndef __CUDA_RANSAC_HOMOGRAPHY_H__
#define __CUDA_RANSAC_HOMOGRAPHY_H__

#include <vector>
#include "DataTypes.h"

using namespace std;

/**
 * NORMALISE_INPUT_POINTS 
 *     normalises the point before calculating the Homography. 
 *     In theory, it should provided better stability. 
 *     Try with and without and see what you get.
 */
#define NORMALISE_INPUT_POINTS // Homohraphy calculation
/**
 * BIAS_RANDOM_SELECTION 
 * uses my variant of RANSAC that uses a bias random number generation, 
 * than than uniform like the vanilla implementation. 
 * The bias random number is based on the score of the matched feature pairs. 
 * Features with better scoring matches will tend to get picked more often than those with lower scores. 
 * This option is useful when the number of iterations is low for the given inliers/outlier ratio.
 */
// #define BIAS_RANDOM_SELECTION // For RANSAC, slightly more intelligent way ..of picking points

void CUDA_RANSAC_Homography(const vector <Point2Df> &src, const vector <Point2Df> &dst, /*const vector <float> &match_score,*/
                            float inlier_threshold, int iterations,
                            /*int *best_inliers, */float *best_H, std::vector <char> *inlier_mask);

#endif
