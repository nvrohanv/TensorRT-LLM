/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "CutlassUtils.h"

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumElts>
inline __device__ void computeMxE4m3SfAndOutputScale(float& outputScale,
                                                     cutlass::float_ue8m0_t& sfOut,
                                                     float const (&input)[NumElts]) {
  // MxE4m3 uses one UE8M0 scale for each group of 32 E4M3 elements.
  int32_t constexpr NumEltsPerSf = 32;
  static_assert(NumEltsPerSf % NumElts == 0, "Not implemented.");

  float localAmax = 0.f;
#pragma unroll
  for (int32_t ii = 0; ii < NumElts; ++ii) {
    localAmax = fmaxf(localAmax, fabsf(input[ii]));
  }

  // Reduce amax across the threads that own the same 32-element output group.
  int32_t constexpr NumThreadsPerSf = NumEltsPerSf / NumElts;
  static_assert(NumThreadsPerSf == 1 || NumThreadsPerSf == 2 || NumThreadsPerSf == 4,
                "Not implemented.");
#pragma unroll
  for (int32_t step = 1; step < NumThreadsPerSf; step *= 2) {
    localAmax = fmaxf(__shfl_xor_sync(uint32_t(-1), localAmax, step), localAmax);
  }

  float const amaxPow2 = trunc_abs_float_to_pow2(localAmax);
  float const sfVal = amaxPow2 * (1.f / 256.f);
  cutlass::Array<float, 1> sfArrayFp32;
  sfArrayFp32[0] = sfVal;
  sfOut = castArray<cutlass::float_ue8m0_t>(sfArrayFp32)[0];
  outputScale = sfVal != 0.f ? scale_rcp_exp_only(sfVal) : 0.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumElts>
inline __device__ void convertAndStoreToGmemAsMxE4m3(char* gmemPtr,
                                                     char* gmemSfPtr,
                                                     float (&input)[NumElts],
                                                     bool isValidStore,
                                                     bool storesSf) {
  static_assert(NumElts == 8, "Not implemented.");

  float outputScale;
  cutlass::float_ue8m0_t sfOut;
  computeMxE4m3SfAndOutputScale(outputScale, sfOut, input);

  float scaled[NumElts];
#pragma unroll
  for (int32_t ii = 0; ii < NumElts; ++ii) {
    scaled[ii] = input[ii] * outputScale;
  }

  if (isValidStore) {
    uint2 output;
    output.x = convert_float4_to_e4m3(scaled[0], scaled[1], scaled[2], scaled[3]);
    output.y = convert_float4_to_e4m3(scaled[4], scaled[5], scaled[6], scaled[7]);
    *reinterpret_cast<uint2*>(gmemPtr) = output;
    if (storesSf) {
      *reinterpret_cast<cutlass::float_ue8m0_t*>(gmemSfPtr) = sfOut;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
