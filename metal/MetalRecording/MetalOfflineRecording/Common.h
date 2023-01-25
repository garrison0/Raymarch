// COMMON SHADER FUNCTIONS (NOISE, MATH, ETC...)
#ifndef Common_h
#define Common_h

float3 LPOS();
float3 LCOL();

float mod289(float x);
float3 mod289(float3 x);
float4 mod289(float4 x);
float4 perm(float4 x);
float noise(float3 p);

float fbm( float3 p );

float3 hash3( float2 p );

float2 hash2( float2 p );

float vnoise( float2 x, float u, float v );

float vnoiseOctaves (float2 x, float u, float v );

float4 permute(float4 x);

float4 taylorInvSqrt(float4 r);

float snoise(float3 v);

float curlNoise( float3 p );

float3 N13(float p);

float linearstep( float s, float e, float v );

float3 rotatePoint(float3 p, float3 n, float theta);

// essentially a bump signal over time
float Saw(float b, float t);

// Triplanar mapping
float tex3D(metal::sampler defaultSampler, metal::texture2d<float, metal::access::sample> tex, float3 pos, float3 nor, float texScale);

// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total.
float3 doBumpMap( metal::sampler defaultSampler, metal::texture2d<float, metal::access::sample> tex, float texScale, float3 p, float3 nor, float bumpfactor);

// procedural coloring trick
float3 palette( float t, float3 a, float3 b, float3 c, float3 d );

metal::float3x3 setCamera( float3 ro, float3 ta, float cr );

#endif /* Common_h */
