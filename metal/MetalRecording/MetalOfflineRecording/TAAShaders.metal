#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

#define VXAA_W 0
#define VXAA_E 1
#define VXAA_N 2
#define VXAA_S 3
#define VXAA_NW 0
#define VXAA_NE 1
#define VXAA_SW 2
#define VXAA_SE 3

float VXAALuma( float3 c )
{
    return dot( c, float3( 0.2126, 0.7152, 0.0722 ) );
}

float VXAALumaDif (float3 c, float3 d)
{
    return abs ( VXAALuma(c) - VXAALuma(d) );
}

float3 VXAAClampHistory( float3 history, float4 currN[4] )
{
    float3 cmin = min( min( currN[0].rgb, currN[1].rgb ), min( currN[2].rgb, currN[3].rgb ) );
    float3 cmax = max( min( currN[0].rgb, currN[1].rgb ), max( currN[2].rgb, currN[3].rgb ) );
    return float3(
        clamp( history.r, cmin.r, cmax.r ),
        clamp( history.g, cmin.g, cmax.g ),
        clamp( history.b, cmin.b, cmax.b )
    );
}

float2 VXAADifferentialBlendWeight (float4 n[4])
{
    float diffNS = VXAALumaDif(n[VXAA_N].rgb, n[VXAA_S].rgb);
    float diffEW = VXAALumaDif(n[VXAA_E].rgb, n[VXAA_W].rgb);
    return (diffNS < diffEW) ? float2(0.5, 0.0) : float2(0.0, 0.5);
}

float4 VXAADifferentialBlend( float4 n[4], float2 w )
{
    float4 c = float4( 0.0 );
    c += ( n[ VXAA_W ] + n[ VXAA_E ] ) * w.x;
    c += ( n[ VXAA_N ] + n[ VXAA_S ] ) * w.y;
    return c;
}

void VXAAUpsampleT4x( thread array<float4, 4>& vtex, float4 current, float4 history, float4 currN[4], float4 histN[4] )
{
    // make the grid
    float4 n1[4], n2[4];
    n1[VXAA_W] = currN[VXAA_W];
    n1[VXAA_E] = current;
    n1[VXAA_N] = histN[VXAA_N];
    n1[VXAA_S] = history;
     
    n2[VXAA_W] = history;
    n2[VXAA_E] = histN[VXAA_E];
    n2[VXAA_N] = current;
    n2[VXAA_S] = currN[VXAA_S];
    
    float4 weights = float4( VXAADifferentialBlendWeight (n1), VXAADifferentialBlendWeight(n2) );
    vtex[VXAA_NW] = history;
    vtex[VXAA_SE] = current;
    vtex[VXAA_SW] = VXAADifferentialBlend( n2, weights.zw );
    vtex[VXAA_NE] = VXAADifferentialBlend( n1, weights.xy );
}

kernel void taa(texture2d<float, access::read> cur [[ texture(0) ]],
                texture2d<float, access::read> prev [[ texture(1) ]],
                texture2d<float, access::write> dest [[ texture(2) ]],
                sampler defaultSampler [[sampler(0)]],
                uint2 gid [[thread_position_in_grid]],
                constant Uniforms &uniforms [[buffer(0)]])
{
    float4 current = cur.read(gid);
    float4 history = prev.read(gid);
    current.a = VXAALuma( current.rgb ); history.a = VXAALuma( history.rgb );
    
    float4 currN[4];
    currN[VXAA_W] = saturate(cur.read( gid + uint2(-1, 0) ));
    currN[VXAA_E] = saturate(cur.read( gid + uint2(1, 0) ));
    currN[VXAA_N] = saturate(cur.read( gid + uint2(0, 1) ));
    currN[VXAA_S] = saturate(cur.read( gid + uint2(0, -1) ));
    currN[VXAA_W].a = VXAALuma( currN[ VXAA_W ].rgb );
    currN[VXAA_E].a = VXAALuma( currN[ VXAA_E ].rgb );
    currN[VXAA_N].a = VXAALuma( currN[ VXAA_N ].rgb );
    currN[VXAA_S].a = VXAALuma( currN[ VXAA_S ].rgb );
    
    float4 histN[4];
    histN[VXAA_W] = saturate(prev.read( gid + uint2(-1, 0) ));
    histN[VXAA_E] = saturate(prev.read( gid + uint2(1, 0) ));
    histN[VXAA_N] = saturate(prev.read( gid + uint2(0, 1) ));
    histN[VXAA_S] = saturate(prev.read( gid + uint2(0, -1) ));
    histN[VXAA_W].a = VXAALuma( histN[ VXAA_W ].rgb );
    histN[VXAA_E].a = VXAALuma( histN[ VXAA_E ].rgb );
    histN[VXAA_N].a = VXAALuma( histN[ VXAA_N ].rgb );
    histN[VXAA_S].a = VXAALuma( histN[ VXAA_S ].rgb );
    history.rgb = VXAAClampHistory( history.rgb, currN );
    
//    float4 vtex[4];
    array<float4, 4> vtex;
    VXAAUpsampleT4x( vtex, current, history, currN, histN );

    // Average all samples.
    float4 col = ( vtex[VXAA_NW] + vtex[VXAA_NE] + vtex[VXAA_SW] + vtex[VXAA_SE] ) * 0.25f;
//    float4 col = current; // <--test the difference
    
    dest.write(col, gid);
}


#define VXAA_TEXTURE_CURRENT iChannel0
#define VXAA_TEXTURE_PREV iChannel1

#define VXAA_TEMPORALEDGE_THRES 0.05
#define VXAA_TEMPORALEDGE_TIME_MIN 0.0000001
#define VXAA_TEMPORALEDGE_TIME_MAX 1.15
#define VXAA_SPATIAL_FLICKER_TIME 2.35
#define VXAA_MORPHOLOGICAL_STRENGTH 0.42
#define VXAA_MORPHOLOGICAL_SHARPEN 0.13

#define VXAA_W 0
#define VXAA_E 1
#define VXAA_N 2
#define VXAA_S 3
#define VXAA_NW 0
#define VXAA_NE 1
#define VXAA_SW 2
#define VXAA_SE 3

float4 pow3( float4 x, float y )
{
    return float4( pow( x.x, y ), pow( x.y, y ), pow( x.z, y ), x.w );
}

float VXAATemporalContrast( float currentLuma, float historyLuma )
{
    float x = saturate( abs( historyLuma - currentLuma ) - VXAA_TEMPORALEDGE_THRES );
    float x2 = x * x, x3 = x2 * x;
    return saturate( 3.082671957671837 * x - 3.9384920634917364 * x2 + 1.8518518518516354 * x3 );
}

float VXAAMorphStrengthShaper( float x )
{
    return 1.3 * x - 0.3 * x * x;
}

float VXAASpatialContrast( float2 spatialLumaMinMax )
{
    float spatialContrast = spatialLumaMinMax.y - spatialLumaMinMax.x;
    return mix( 0.0f, 1.0f, spatialContrast );
}

float VXAATemporalFilterAlpha( float fpsRcp, float convergenceTime )
{
    return exp( -fpsRcp / convergenceTime );
}

float4 VXAASharpen( float4 history, float4 histN[4] )
{
    float4 nh = histN[VXAA_NW] + histN[VXAA_NE] + histN[VXAA_SW] + histN[VXAA_SE];
    return mix( history, history * 5.0f - nh, VXAA_MORPHOLOGICAL_SHARPEN );
}

float4 VXAAMorphological( float2 uv, float4 current, float4 currN[4], float strength, float2 res, sampler defaultSampler, texture2d<float, access::sample> cur)
{
    if ( strength < 0.1f ) return current;
    float lumaNW = currN[VXAA_NW].a, lumaNE = currN[VXAA_NE].a,
        lumaSW = currN[VXAA_SW].a, lumaSE = currN[VXAA_SE].a;
    lumaNE += 0.0025;

    float2 dir;
    dir.x = ( lumaSW - lumaNE ) + ( lumaSE - lumaNW );
    dir.y = ( lumaSW - lumaNE ) - ( lumaSE - lumaNW );
    float2 dirN = normalize( dir );
    
    float4 n1 = cur.sample(defaultSampler, uv - dirN * strength / res );
    float4 p1 = cur.sample(defaultSampler, uv + dirN * strength / res );
    return ( n1 + p1 ) * 0.5;
}

float4 VXAAFilmic( float2 uv, float4 current, float4 history, float4 currN[4], float4 histN[4], float time, float2 res, sampler defaultSampler, texture2d<float, access::sample> cur )
{
    // Temporal contrast weight.
    float temporalContrastWeight = VXAATemporalContrast( current.a, history.a );

    // Spatial contrast weight.
    float2 spatialLumaMinMaxC = float2(
        min( min( currN[0].a, currN[1].a ), min( currN[2].a, currN[3].a ) ),
        max( max( currN[0].a, currN[1].a ), max( currN[2].a, currN[3].a ) )
    );
    float2 spatialLumaMinMaxH = float2(
        min( min( histN[0].a, histN[1].a ), min( histN[2].a, histN[3].a ) ),
        max( max( histN[0].a, histN[1].a ), max( histN[2].a, histN[3].a ) )
    );
    float spatialContrastWeightC = VXAASpatialContrast( spatialLumaMinMaxC );
    float spatialContrastWeightH = VXAASpatialContrast( spatialLumaMinMaxH );
    float spatialContrastWeight = abs( spatialContrastWeightC - spatialContrastWeightH );
    
    // Evaluate convergence time from weights.
    float convergenceTime = mix( VXAA_TEMPORALEDGE_TIME_MIN, VXAA_TEMPORALEDGE_TIME_MAX, temporalContrastWeight );
    convergenceTime = mix( convergenceTime, VXAA_SPATIAL_FLICKER_TIME, spatialContrastWeight );
    float alpha = VXAATemporalFilterAlpha( time, convergenceTime );
    
    // Apply morpholigical AA filter and sharpen.
    float strength = VXAAMorphStrengthShaper( spatialContrastWeightC * 4.0 ) * VXAA_MORPHOLOGICAL_STRENGTH;
    current = VXAAMorphological( uv, current, currN, strength, res, defaultSampler, cur );
    current = VXAASharpen( current, currN );
    
    // Clamp history to neighbourhood, and apply filmic blend.
    history.rgb = VXAAClampHistory( history.rgb, currN );
    current = mix( current, history, alpha );
    return current;
}

kernel void filmicSMAA(texture2d<float, access::sample> cur [[ texture(0) ]],
                       texture2d<float, access::sample> prev [[ texture(1) ]],
                       texture2d<float, access::write> dest [[ texture(2) ]],
                       sampler defaultSampler [[sampler(0)]],
                       uint2 gid [[thread_position_in_grid]],
                       constant Uniforms &uniforms [[buffer(0)]])
{
    float2 res = float2(dest.get_width(), dest.get_height());
    float2 uv = float2(gid) / res;
    
    // Sample scene and neighbourhood.
    
    float4 current = clamp( float4( cur.sample(defaultSampler, uv ).rgb, 1.0 ), float4( 0.0f ), float4( 1.0f ) );
    float4 history = clamp( float4( prev.sample(defaultSampler, uv ).rgb, 1.0 ), float4( 0.0f ), float4( 1.0f ) );
    current.a = VXAALuma( current.rgb ); history.a = VXAALuma( history.rgb );
    
    float4 currN[4];
    currN[VXAA_NW] = clamp( cur.sample(defaultSampler, uv + 0.6f * float2( -1.0f,  1.0f ) / res), float4( 0.0f ), float4( 1.0f ) );
    currN[VXAA_NE] = clamp( cur.sample(defaultSampler, uv + 0.6f * float2(  1.0f,  1.0f ) / res), float4( 0.0f ), float4( 1.0f ) );
    currN[VXAA_SW] = clamp( cur.sample(defaultSampler, uv + 0.6f * float2( -1.0f, -1.0f ) / res), float4( 0.0f ), float4( 1.0f ) );
    currN[VXAA_SE] = clamp( cur.sample(defaultSampler, uv + 0.6f * float2(  1.0f, -1.0f ) / res), float4( 0.0f ), float4( 1.0f ) );
    currN[VXAA_NW].a = VXAALuma( currN[VXAA_NW].rgb );
    currN[VXAA_NE].a = VXAALuma( currN[VXAA_NE].rgb );
    currN[VXAA_SW].a = VXAALuma( currN[VXAA_SW].rgb );
    currN[VXAA_SE].a = VXAALuma( currN[VXAA_SE].rgb );
    
    float4 histN[4];
    histN[VXAA_NW] = clamp( prev.sample(defaultSampler, uv + 0.6f * float2( -1.0f,  1.0f ) / res ), float4( 0.0f ), float4( 1.0f ) );
    histN[VXAA_NE] = clamp( prev.sample(defaultSampler, uv + 0.6f * float2(  1.0f,  1.0f ) / res ), float4( 0.0f ), float4( 1.0f ) );
    histN[VXAA_SW] = clamp( prev.sample(defaultSampler, uv + 0.6f * float2( -1.0f, -1.0f ) / res ), float4( 0.0f ), float4( 1.0f ) );
    histN[VXAA_SE] = clamp( prev.sample(defaultSampler, uv + 0.6f * float2(  1.0f, -1.0f ) / res ), float4( 0.0f ), float4( 1.0f ) );
    histN[VXAA_NW].a = VXAALuma( histN[VXAA_NW].rgb );
    histN[VXAA_NE].a = VXAALuma( histN[VXAA_NE].rgb );
    histN[VXAA_SW].a = VXAALuma( histN[VXAA_SW].rgb );
    histN[VXAA_SE].a = VXAALuma( histN[VXAA_SE].rgb );
    
    
    // Filmic pass.
    dest.write(VXAAFilmic( uv, current, history, currN, histN, uniforms.time, res, defaultSampler, cur ), gid);
}
