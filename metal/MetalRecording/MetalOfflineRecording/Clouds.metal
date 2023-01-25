
#include <metal_stdlib>
using namespace metal;

#include "Common.h"
#include "ShaderTypes.h"

float2 CLOUDEDGES() {
    float baseline = 4.0;
    return baseline+float2(-1.5, 3.5);
}

float CLOUDGRADIENT(float3 pos) {
    float2 cloudEdges = CLOUDEDGES();
    return (smoothstep(cloudEdges.x, cloudEdges.x + 1.0, pos.y) - smoothstep(cloudEdges.y - 1.0, cloudEdges.y, pos.y));
}

float CLOUDNORY(float3 pos) {
    return linearstep(CLOUDEDGES().x, CLOUDEDGES().y, pos.y);
}

float3 evaluateLight( float3 pos )
{
//    float norY = 0.01+CLOUDNORY(pos);
    float3 lightPos = 20*normalize(float3(0.0,0.5,-0.6));
    float3 lightCol = float3(exp(2.15)) * LCOL();
    float lamp = smoothstep(0.9, 1.0, (dot(float3(0,-1.0,0), normalize(pos - lightPos))));
    return lamp * lightCol;
}

float remap(float v, float s, float e) {
    return (v - s) / max((e - s), 0.0001);
}

float cloudMapBase(float3 p, float norY, float time) {
    float3 uv = p * (0.04) * float3(1,2.0,1);
    
    float r = fbm(1.5*uv + 0.8*fbm(uv + 0.017*time*float3(1.0,0.5,0.75)) + 0.01*float3(1.0,0.5,0.75)*time);

    return r;
}

void getParticipatingMedia(thread float& sigmaS, thread float& sigmaE, float3 pos, float time)
{
    time *= 1.5;
    pos += time * 0.075 * float3(0.3,-0.06,0.15);
    float3 ps = pos;
    float norY = CLOUDNORY(pos);
    
//    float m = cloudMapBase(ps, norY, time);
//    float yGradient = CLOUDGRADIENT(pos);
//    m *= yGradient;
//
//    float dstrength = smoothstep(1., 0.5, m);
//
//    // erode with detail
//    if(dstrength > 0.) {
//        float second = fbm(0.45*pos + 0.2*snoise(time*0.077+0.8*pos) + 0.6*fbm(0.4*pos - time*0.047));
//        m -= second * dstrength * 0.35;
//    }
//
//    float cloudDensity = 0.25;
//    m = cloudDensity*smoothstep( 0.0, 0.1, m - 0.45);

    float constantFog = 0.001;

    sigmaS = constantFog;// + m;

    float sigmaA = 0.0; // absorb
    sigmaE = max(0.00001, sigmaA + sigmaS); // to avoid division by zero extinction
}

// henyey-greenstein
float phg(float ang, float g) {
    float pi = 3.141592653;
    float g2 = pow(g, 2.0);
    return (1. - g2) / (4.*pi*pow(1.0 + g2 - 2.*g*cos(ang), 1.5));
}

float phaseFunction( float ang )
{
    return mix(phg(ang, 0.0), phg(ang, 0.75), 0.5);
}

float volumetricShadow(float3 from, float3 to, float time)
{
    float dd = 1.4;
    float d = dd * .5;
    float sigmaS = 0.0;
    float sigmaE = 0.0;
    float shadow = 1.0;
    float3 rd = normalize(from - to);
    for(float s=0.5; s<16.-0.5; s++) {
        float3 pos = from + rd * d;

        getParticipatingMedia(sigmaS, sigmaE, pos, time);
        shadow *= exp(-sigmaE * dd);

        dd *= 1.5;
        d += dd;
    }
    return shadow;
}

float4 marchClouds (float3 ro, float3 rd, float time, float intersectT){
    float transmittance = 1.0;
    float3 scatteredLight = float3(0.0, 0.0, 0.0);
    
    int NUM_STEPS = 50;
    
    float tmin = 0.0;
    float tmax = 75.0;
    float t = tmin;
    
    float sigmaS = 0.0;
    float sigmaE = 0.0;
    
    float3 lightPos = 80.0*LPOS();
    
    float d = tmax / float(NUM_STEPS);
    
    for( int i = 0; i < NUM_STEPS; i++) {
        if ( t > intersectT ) // we hit some object, now we're behind it
        {
            break;
        }
        
        float3 p = ro + rd*t;
        
        getParticipatingMedia(sigmaS, sigmaE, p, time);

        int NUM_OCTS = 1;
        for (float j = 0.; j <= float(NUM_OCTS); j++ )
        {
            float sigmaS2 = sigmaS * pow(0.5, j);
            float sigmaE2 = sigmaE * pow(0.5, j);

            float ang = acos(dot(normalize(lightPos*float3(1,-1,1)), rd));
            ang *= pow(0.5,j);

            float invNorY = 0.75 + 0.25*smoothstep(CLOUDEDGES().x, CLOUDEDGES().y, p.y);
            float3 ambient = invNorY * float3(0.85,0.9,0.99) * pow(0.5,j+1);
            ambient = 0.75 * pow(ambient, 2.0);

            float3 S = sigmaS2 * (ambient + evaluateLight(p) * phaseFunction(ang) * volumetricShadow(p,lightPos,time));
            float3 Sint = (S - S * exp(-sigmaE2 * d)) / sigmaE2;
            scatteredLight += transmittance * Sint;
        }

        transmittance *= exp(-sigmaE * d);
    
        t += d;
    }

    // return scattering, transmittance (inout equivalent scatTran)
    float4 scatTrans = float4(scatteredLight, transmittance);
    return scatTrans;
}

kernel void computeClouds(texture2d<float, access::write> output [[texture(0)]],
                          texture2d<float, access::sample> worldRender [[texture(1)]],
                          sampler defaultSampler [[sampler(0)]],
                          uint2 gid [[thread_position_in_grid]],
                          constant Uniforms &uniforms [[buffer(0)]])
{
    float change = 48.2; // 20 was good
    float t = uniforms.time + 30.0;
    float3 ro = float3( 0.0, change+4.5, (t/25.0) + 1.25);
    float3 ta = float3( 0.0, change+4.41, 0.5);
    float3x3 ca = setCamera(ro, ta, 0.0);

    int width = output.get_width();
    int height = output.get_height();
    float2 res = float2(width, height);
    float aspect = res.x/res.y;
    float2 uv = float2(gid) / res;
    float2 worldUV = uv;
    
//    worldUV = worldUV - 1/;
    uv = uv * 2.0 - 1.0;
    uv.y *= -1.0;
    
    // jitter subpixel position each frame
//    float2 o = float2(0.0);
    float2 o = (hash2(uv+t*0.1) / res);
    float2 p = float2(aspect, 1.0) * (uv+o);
    
    float3 rd = ca * normalize( float3(p, 2.2) );
    
    float2 worldPixel = float2(1.0) / float2(worldRender.get_width(), worldRender.get_height());
    float intersect = worldRender.sample(defaultSampler, worldUV).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + worldPixel*float2(1.0,0.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + worldPixel*float2(-1.0,0.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + worldPixel*float2(0.0,1.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + worldPixel*float2(0.0,-1.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + 0.6*worldPixel*float2(1.0,1.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + 0.6*worldPixel*float2(-1.0,1.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + 0.6*worldPixel*float2(1.0,-1.0)).a * 0.1111;
    intersect += worldRender.sample(defaultSampler, worldUV + 0.6*worldPixel*float2(-1.0,-1.0)).a * 0.1111;
    intersect *= 2000.0; // max length for ray in raycast
    float4 scatTrans = marchClouds(ro, rd, t, intersect);
    
    //scatTrans.xyz = pow(scatTrans.xyz, float3(0.4545)); <-- do this in the composite step it if looks strange
    scatTrans = saturate(scatTrans);

    // dithering
    //total += (1.0/100.0) * sin(uv.x*944.111)*sin(uv.y*843.32);
    output.write(scatTrans, gid);
}
