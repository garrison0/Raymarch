// SHARED TYPES BETWEEN MSL AND SWIFT

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef struct
{
    float time;
    simd_float2 resolution;
} Uniforms;

#endif

