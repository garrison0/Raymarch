
#include <metal_stdlib>
using namespace metal;
#include "ShaderTypes.h"

kernel void compositeTextures(texture2d<float, access::sample> clouds [[ texture(0) ]],
                texture2d<float, access::sample> world [[ texture(1) ]],
                texture2d<float, access::write> dest [[ texture(2) ]],
                sampler defaultSampler [[sampler(0)]],
                constant Uniforms &uniforms [[buffer(0)]],
                uint2 gid [[thread_position_in_grid]])
{
//    int width = dest.get_width();
//    int height = dest.get_height();
    float2 uv = float2(gid) / uniforms.resolution;
    
    float3 worldCol = world.sample(defaultSampler, uv).rgb;
    float4 scatTrans = clouds.sample(defaultSampler, uv);
    float3 colSoftEdge = worldCol * scatTrans.w + scatTrans.xyz;
    float3 col = mix(scatTrans.xyz, worldCol, scatTrans.w); // SCATTRANS.W 'dirty cloud' effect
//    col = 0.5 * col + 0.5 * colSoftEdge;
    col = colSoftEdge;
                            // alternatively: col = col * scatTrans.w + scatTrans.xyz;
//    float3 col = scatTrans.xyz + worldCol * scatTrans.w;
    dest.write(float4(col, 1.0), gid);
//    dest.write(float4(worldCol, 1.0), gid);
}

