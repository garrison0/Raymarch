#include <metal_stdlib>
#include "ShaderTypes.h"
#include "Common.h"
#include "SDFs.h"

#define program 4

using namespace metal;

float3 transformPendulumPoint(float3 p, float time) {
    time = time * 0.48;
    p.y += 15.0;
    p.x += 20.0;
    float cx = fmod((p.x) + 2.0, 4.0) - 2.0;
    float cy = fmod((p.y) + 2.0, 4.0) - 2.0;
    float3 q = float3(cx,cy,p.z);
    float2 iq = float2(floor((p.x+2.0)/4.0), floor((p.y+2.0)/4.0));
    
    float qNoise = snoise(float3(2.0*time,1.0*time,0) + float3(iq.x,iq.y,0.1));
    return rotatePoint(q + float3(0,-0.35,-0.15), float3(1,0,0), qNoise*3.141593653/18);
}


float3 twistTransform( float3 p, float time )
{
    // rotate
    p = rotatePoint(p, normalize(float3(0.0,1,0.1)), 0.5*time);
    p.y *= -1.0;
    float scale = 1.0 + (0.5+0.5*sin(0.5*time));
    float scale2 = (0.5+0.5*cos(0.2223 + pow(cos(p.y + 0.5*time), 3.0)));
    p.xz *= 1.0 + scale*( 1.0 - (smoothstep(-1.75, -0.75, p.y) - smoothstep(-0.75, 0.5, p.y)) ); //-1.75, 0.5
//    p.x *= (0.1 + saturate(smoothstep(0, 0.5, p.y) - smoothstep(0.5, 1.0, p.y)));
    
    // twist
    float k = 3.0 + 2*sin(0.75*time);
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    float2x2 rot = float2x2(c, -s, s, c);
    
    p.xz = rot * p.xz;
    
    p.y *= (1.25 - 0.1*scale2);
    
    return p;
}

float2 mapChair(float3 p, float time, float2 res )
{
    float2 worldRes = res;
    p.y += 1.0;
    
    float3 u = p; // for two back supports (x = abs(x))
    float3 v = p; // for back join supports
    float3 q = p; // for legs
    
    // chair
    // seat
    res = float2(1e10, 0.0);
    float d = sdCappedCylinderVertical(p + float3(0,0.01,0), 1.02, 0.004) - 0.03;
    res = opU(res, float2(d, 15.0));
    
    p.x = abs(p.x);
    p.z = abs(p.z);

    // legs
    float r = 1.15 - 0.16 * p.y;
    float r2 = 1.15;

    float x = r*cos(3.141592653/4.0);
    float y = r*sin(3.141592653/4.0);
    float x2 = r2*cos(3.141592653/4.0);
    float y2 = r2*sin(3.141592653/4.0);
    
    float3 legLocation = float3( x, 0.0, y );
    d = sdCappedCylinderVertical(p - (float3(0,-0.52,0)+0.935*legLocation), 0.05, 0.325) - 0.01;
    
    //bars connecting legs under the chair
    float d2 = sdCapsule(p, float3(0,-0.1,0), (float3(0, -0.1, 0) + 0.9*legLocation)).x - 0.0365; //.075

    q.x = abs(q.x);
    q.z = abs(q.z);
        
    q -= (0.85*legLocation + float3(0,-0.2,0));
    float s = sin(- 3.141592653/4.0);
    float c = cos(- 3.141592653/4.0);
    
    float3x3 ry = float3x3(c, 0, s,
                           0, 1, 0,
                           -s, 0, c);
    
    s = sin(- 3.141592653/4.0);
    c = cos(- 3.141592653/4.0);
    float3x3 rz = float3x3(c, -s, 0,
                           s, c, 0,
                           0, 0, 1);
    
    float3x3 rot = rz * ry;
    q = rot * q;
    
    // joining chair seat support with legs
    float d3 = sdCappedTorus(q, float2(0.3,0.3), 0.1, 0.035 + 0.025) - 0.00;
    
    float2 res2 = opSmoothU( float2(d2*0.1, 5.0), float2(d, 5.0), 0.06);
    res2 = opSmoothU(res2, float2(d3*0.1, 5.0), 0.0);
    res = opU(res, res2);
    
    // back leg support
    d = sdCapsule(v - float3(0, -0.35, y*0.95), float3(-x*0.95,0,0), float3(x*0.95,0,0)).x - 0.0125;
    
    res = opSmoothU(res, float2(d, 10.0), 0.04);
    
    // back chair rest supports
    u.x = abs(u.x);
    
    float3 backStart = u - float3(x2*0.95 + 0.015*u.y, -0.1, y2*0.95 + 0.075*u.y);
    d = sdCapsule(backStart, float3(0), float3(0,0.35,0)).x - 0.025;
    d2 = sdCapsule(backStart - float3(0,0.35,0), float3(0), float3(x2*0.04, 0.81, y2*0.4)).x - 0.025;
    
    res2 = opSmoothU(float2(d2, 5.0), float2(d, 5.0), 0.01);
    res = opSmoothU(res, res2, 0.04);
    
    //  two supporting beam + thick wood support at time
    float vx = (1.2 + clamp(v.x, -1.15, 1.15)) / 2.4;
    d = sdCapsule( v - float3( 0, 1.125, 0.5*pow(sin(3.14159 * vx),0.5) + 0.853), float3( -1.15, 0, 0), float3(1.15, 0, 0)).x - 0.0625;
    res = opSmoothU(res, float2(d, 10.0), 0.005);
    
    vx = (.8 + clamp(v.x, -.8, .8)) / 1.6;
    d = sdCapsule( v - float3( 0, 0.775, 0.175*pow(sin(3.14159 * vx),0.5) + 1.053), float3( -0.8, 0, 0), float3(0.8, 0, 0)).x - 0.0125;
    res = opSmoothU(res, float2(d, 10.0), 0.04);
    
    vx = (.775 + clamp(v.x, -.775, .775)) / 1.55;
    d = sdCapsule( v - float3( 0, 0.525, 0.175*pow(sin(3.14159 * vx),0.5) + 0.923), float3( -0.775, 0, 0), float3(0.775, 0, 0)).x - 0.0125;
    res = opSmoothU(res, float2(d, 10.0), 0.04);
    
    program == 5 ? res.y = 1.0 : 0.0;
    worldRes = opSmoothU(res, worldRes, program == 5 ? 2.75 : 0.005);
    return worldRes;
}

float3 GerstnerWave(float2 coord, float wavelength, float steepness, float2 direction, float time)
{
    const float gravitationalConst = 9.81;
    
    float3 gerstner;
    float k = 2.0 * 3.141592653 / wavelength;
    float c = sqrt(gravitationalConst / k);
    float a = steepness / k;
    float2 dir = normalize(direction);
    float f = k * (dot(dir, coord.xy) - c * time * 2.0);
    
    gerstner.x += dir.x * (a * cos(f));
    gerstner.y = a * sin(f);
    gerstner.z += dir.y * (a * cos(f));
    
    return gerstner;
}

float2 fanOut(float2 coords, float divisions) {
    float k = 2*3.141592653 / divisions;
    float angle = k * -floor(atan2(coords.y, coords.x) / k);
    float c = cos(angle), s = sin(angle);
    float2x2 rot = float2x2(c, -s, s, c);
    coords *= rot;
    return coords;
}

float2 map(float3 p, float time )  {
    float2 res = float2(1e10, 0.0);

    p -= float3(0,3.8,-3.25);
//
    res = opU(res, float2(sdBox(p + float3(0,8,0), float3(1000.0, 1.0, 1000.0)), 1.0));

    float cx = fmod((p.x) + 2.0, 4.0) - 2.0;
    float cy = fmod((p.y) + 2.0, 4.0) - 2.0;
    float3 q = float3(cx,cy,p.z);
    float2 iq = float2(floor((p.x+2.0)/4.0), floor((p.y+2.0)/4.0));
//    //float qNoise = snoise(float3(2.0*time,1.0*time,0) + float3(iq.x,iq.y,0.1));
////    float3 q = p;

    // modelwide transform (separate func, use for texturing)
    float3 wavesX = float3(0);
    float3 wavesY = float3(0);
    float3 wavesZ = float3(0);
    float s = 1.0;
    float offset = 0.0;
    if (program == 1) {
        p = twistTransform(p, time);
    } else if (program == 2) {
        p = rotatePoint(p, normalize(float3(0.0,1,0.0)), 0.0);
        p.y *= -1.0;

        wavesX += GerstnerWave(p.y * 20.0, 20.0, 1.0, float2(1, 1), time);
        wavesX += GerstnerWave(p.y * 30.0, 31.0, 1.5, float2(1, 0.6), time);
        wavesX += GerstnerWave(p.y * 40.0, 58.0, 0.5, float2(1, 1.3), time);
        wavesX += GerstnerWave(p.y * 50.0, 26.0, 2.0, float2(0.7, 1.0), time);
        wavesX += GerstnerWave(p.y * 20.0, 22.0, 1.8, float2(0.8, 0.6), time);

        wavesY += GerstnerWave(p.z * 40.0, 20.0, 1.0, float2(1, 1), time);
        wavesY += GerstnerWave(p.z * 40.0, 31.0, 0.6, float2(1, 0.6), time);
        wavesY += GerstnerWave(p.z * 50.0, 58.0, 0.5, float2(1, 1.3), time);
        wavesY += GerstnerWave(p.z * 60.0, 26.0, 0.5, float2(0.7, 1.0), time);
        wavesY += GerstnerWave(p.z * 55.0, 22.0, 1.2, float2(0.8, 0.6), time);

        wavesZ += GerstnerWave(p.x * 25.0, 20.0, 1.0, float2(1, 1), time);
        wavesZ += GerstnerWave(p.x * 15.0, 31.0, 1.0, float2(1, 0.6), time);
        wavesZ += GerstnerWave(p.x * 22.0, 58.0, 1.0, float2(1, 1.2), time);
        wavesZ += GerstnerWave(p.x * 50.0, 26.0, 1.0, float2(0.88, 1.0), time);
        wavesZ += GerstnerWave(p.x * 20.0, 22.0, 1.0, float2(0.8, 0.7), time);

        wavesX *= 0.015; wavesY *= 0.04; wavesZ *= 0.0;

        float3 amnts = float3(1.0);
        amnts.x = fbm( float3(time, p.y, p.z) );
        amnts.z = fbm( float3(p.x, p.y, time) );
        amnts.y = fbm( float3(p.x, time, p.z) );

        amnts = normalize(amnts);

        p.x = p.x - wavesX.y * 0.2 * amnts.x;
        p.y = p.y - wavesY.y * 0.2 * amnts.y;
        p.z = p.z - wavesZ.y * 0.2 * amnts.z;
    } else if (program == 3) {
        // kifs..
        // it would be cool if when it returned to the box + chair table thing
        time *= 0.2;
        p = rotatePoint(p, normalize(float3(0.0,1,0.0)), 0.0);
        p.y *= -1.0;
//        offset = 0.0;
        float3x3 kifsRot = float3x3(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0); //all 1
//        offset = -0.25 - 0.15*(0.5+0.5*sin(time)); // nice for box!
        offset = -3.5 + 3.1*(0.5+0.5*sin(sin(time)));
        
        float a = sin(1.54*time) * 1.25 + 1.25; // nice for box
//        float a = sin(time) * 2.25 + 2.25;
        float s = sin(a), c = cos(a);
        kifsRot = kifsRot * float3x3(c, -s, 0, s, c, 0, 0, 0, 1);
        kifsRot = kifsRot * float3x3(1, 0, 0, 0, c, -s, 0, s, c);
        kifsRot = kifsRot * float3x3(c, 0, s, 0, 1, 0, -s, 0, c);
        
        int ITERS = 12; // 12 / 4! 8!
        for (int i = 0; i < ITERS; ++i) { //
            p.xz = fanOut(p.xz, 5.0); // above 6 is meh // 1.0 was really elegant
            p.xz = abs(p.xz); //just p
            p += offset / (ITERS);
//            p += (offset/(ITERS))/(max(1,i)); // 0.5 * ITERS
//            p -= offset;
//            p -= 0.1;
//            p.z -= 0.25;
            
            p = kifsRot * p;
            s *= 2.0;
        }
    } else if (program == 4) {
        // fog + light + infinite reveal
        p = rotatePoint(p, normalize(float3(0.0,1,0.0)), 0.0);
        p.y *= -1.0;
        
        
    } else if (program == 5) {
        // uv effects (come out of the floor
        p.y += 4.0 - 0.1*time;
    }
    
//    float d = sdBox(p, float3(0.01));
//    d *= 0.8;
//    res = opU(res, float2(d, 5.0));
//    res = float2(d, 10.0);
//    sdBox(p * s, vec3(s / 17.)) / s
//    offset.z -= s/2.0;
    // / 17 for 4, 8.
    if (program == 3) {
        res = opSmoothU(res, float2(sdBox(p * s, float3(s/22.) / (s) ), 5.0), 0.01);
    }
    float2 otherRes = mapChair(p * s, time, res);
    otherRes.x /= 1.5;
    res = opSmoothU(res, otherRes, 0.2);
    
//    res.x *= 0.5;

//    res.x *= 0.5;
    
    return res;
}

//float2 map(float3 p, float time )  {
//    float2 res = float2(1e10, -1.0);
//
//    res = opU(res, float2(sdBox(p+float3(0,37.5,0), float3(1000., 1.0, 1000.)), 45.));
//
//    float3 pOrig = p;
//    float3 p2 = p;
//
//    p2.y += 15.0;
//    p2.x += 20.0;
//
//    float cx = fmod((p2.x) + 2.0, 4.0) - 2.0;
//    float cy = fmod((p2.y) + 2.0, 4.0) - 2.0;
//    float3 q = float3(cx,cy,p2.z);
//
//    // clock
//    res = opU(res, float2(sdBox(q - float3(0,0.3,0.0), float3(0.4, 0.8, 0.2)) - 0.025, 10.0));
//    res.x = opSmoothSubtraction(sdBox(q - float3(0,0.25,0.1), float3(0.25, 0.75, 0.15)) - 0.025, res.x, 0.1);
//    res = opSmoothU(res, float2(sdExtrudedTriangle(q + float3(0.0,0.685,0.15), float2(0.22, 0.175), 0.005) - 0.025, 15.0), 0.05);
//    float3 qRot = rotatePoint(q + float3(0.18, 0.635, 0.0), float3(1,0,0), 3.141593653/3.5);
//    float2 resBot = float2(sdBox(qRot, float3(0.035, 0.245, 0.2)) - 0.025, 15.0);
//    qRot = rotatePoint(q + float3(-0.18, 0.635, 0.0), float3(1,0,0), -3.141593653/3.5);
//    resBot = opSmoothU(resBot, float2(sdBox(qRot, float3(0.035, 0.245, 0.2)) - 0.025, 15.0), 0.125); //.045
//    res = opSmoothU(res, resBot, 0.025);
//    res = opSmoothU(res, float2(sdExtrudedOctogon(q - float3(0,0.75,0.165), 0.68, 0.025) - 0.0175, 15.0), 0.125);
//
//    res.x = opSmoothSubtraction(sdCappedCylinder(q - float3(0,0.75,1.0275), float3(0), float3(0,0,-0.85), 0.5), res.x, 0.05);
//
//    // glass
//    float3 qGla = q - float3(0,0.75,-0.92);
//    res = opSmoothU(res, float2(max(-sdBox(qGla + float3(0,0,1.0)*0.7, 0.6*float3(3.0, 3.0, 2.85)), sdfSphere(qGla + float3(0,0,0.005), 0.6*2.0)), 30.5), 0.025);
//
//    // clock pendulum
//    qRot = transformPendulumPoint(pOrig, time);
//    float2 d2 = float2(sdBox(qRot - float3(0,0.35,0), float3(0.02, 0.35, 0.00004)), 5.0);
//    d2 = opSmoothU(d2, float2(sdCappedCylinder(qRot - float3(0,0.7,0.0), float3(0), float3(0.,0.,0.00015), 0.12)-0.0025, 5.0), 0.004);
//
//    res = opU(res, d2);
//
//    return res;
//}
//
//float mapGlass(float3 p, float time) {
//    p.y += 15.0;
//    p.x += 20.0;
//    float cx = fmod((p.x) + 2.0, 4.0) - 2.0;
//    float cy = fmod((p.y) + 2.0, 4.0) - 2.0;
//    float3 q = float3(cx,cy,p.z);
//    return sdExtrudedOctogon(q - float3(0,0.75,0.175), 0.6, 0.02);
//}


// since they're map dependent, leave these functions outside of Common
float2 raycast (float3 ro, float3 rd, float time){
    float2 res = float2(-1.0,-1.0);

    float tmin = 0.021;
    float tmax = 2000.0;
    
    float t = tmin;
    float eps = t*0.0001;
//    float eps = 0.0001;

    float2 h = float2(0);
    int NUM_STEPS = 2428;
    
    for( int i = 0; i < NUM_STEPS && t < tmax; i++) {
        float3 p = ro + rd*t;
        h = map( p, time );

        if( h.x < eps ) {
            res = float2(t, h.y);
            break;
        }
            
        t += h.x/2.0;
           
        eps = t*0.0000225;
    }
    
    res.x = t; // send over the distance even if we didn't hit (i.e. no material id)
    return res;
}

float3 calcNormal( float3 p, float time )
{
    const float eps = 0.003;
    const float2 h = float2(eps,0);
    return normalize( float3(map(p+h.xyy, time).x - map(p-h.xyy, time).x,
                        map(p+h.yxy, time).x - map(p-h.yxy, time).x,
                        map(p+h.yyx, time).x - map(p-h.yyx, time).x ) );
}

float calcAO( float3 pos, float3 nor, float time )
{
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.01 + 0.12*float(i)/4.0;
        float d = map( pos + h*nor, time ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
        if( occ>0.35 ) break;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}

float calcSoftshadow( float3 ro, float3 rd, float mint, float maxt, float k, float time )
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t<maxt; )
    {
        float h = map(ro + rd*t, time).x;
        if( h<0.0005 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h/4.5;
    }
    return res;
}

float waterMap (float2 uv, float time)
{
    return pow( fbm ( float3(0.15 * uv, time * 0.22) ), 2.0);
}

float3 waterBumpMap ( float3 pos, float time )
{
    float3 normal = float3(0,0.5,0);
    
    float bump_scale = 0.25 * smoothstep(600.0, 0.0, abs(pos.z));
    float eps = 0.001;
    float2 dx = float2(eps, 0.0);
    float2 dz = float2(0.0, eps);
    normal.x -= bump_scale * (waterMap(pos.xz + dx, time) - waterMap(pos.xz - dx, time)) / (2.0*eps);
    normal.z -= bump_scale * (waterMap(pos.xz + dz, time) - waterMap(pos.xz - dz, time)) / (2.0*eps);
    return normalize(normal);
}

float3 skyColor(float3 pos, float3 rd) {
//    rd.y -= 0.2;
    float3 col = 0.95*float3(0.52, 0.8, 0.98);
    float sd = pow( saturate(dot(rd, float3(1,-1,1)*LPOS())) , 4.0);
//    return float3(sd);
    col = mix(col, float3(1.0), sd * exp( -abs( -1.0*sd ) ) ); //bright sun
    col = mix(col, float3(0.7,0.7,0.9), exp(-5.0*rd.y)); // darken horizon
    col = mix(col, float3(0.69,0.6,0.8), exp(-20.0*(rd.y+0.04)));
//    col = sd * col;
    
    float t = (1000.0-pos.y)/rd.y;
    float3 p = pos + t * rd;
    col *= LCOL();
    
    if ( t > 0.0 ) {
        col = mix(col, float3(1.0), smoothstep(17500., 000., abs(p.z)) * vnoiseOctaves(p.xz * 0.000969, 1.0, 1.0));
    }
    
    col = pow(col, 2.0);
    return col;
}

float4 render(float3 ro, float3 rd, float3 rdx, float3 rdy, float time, array<texture2d<float, access::sample>, 20> ground, texturecube<float> cubemap, sampler defaultSampler)
{
    float3 col = float3(0.8);
    
    float2 res = raycast(ro,rd,time); // traceScene equivalent

    float t = res.x;
    float m = res.y;

    float3 pos = ro + rd*t;
    float3 nor = calcNormal(pos, time);
    
//    float texIndex = 1.0+8.0;
//    float texScale = 0.75;
//    if (m > 0. && m < 10.) {
//        texIndex = 1.0+4.0;
//        texScale = 1.0;
//    }

    float texIndex = 0.0;
    float texScale = 0.05;
    if (m > 2.) {
        if ( m < 10.) {
            texIndex = 4.0;
            texScale = 0.2;
        } else if (m < 20. ) {
            texIndex = 12.0;
            texScale = 0.2;
        }
    }

    // bump map the normal
    if (m < 20.) {
        float scale = 0.0;
        if (m < 2.) {
            scale = 0.05;
        } else if ( m > 10. ) {
            scale = 0.004;
        }
        nor = doBumpMap(defaultSampler, ground[texIndex+1.0], texScale, pos, nor, scale);
    }
//    if (m < 20.) {
//        nor = doBumpMap(defaultSampler, ground[texIndex+1.0], texScale, pos, nor, m > 10 ? 0.006 : 0.0);
//    }
    
    float3 material = float3(0);
    float3 f0 = float3(0.05);
    float a = 1.0;
    float3 hNor = float3(0.0);
    float rou = 0.0;
    
    // MATERIALS
    // wood
    if (m > 0. && m < 30.) {
        f0 = float3(.04);
        if (m > 2.0 && m < 10.0) {
            f0 = float3(0.99, 0.97, 0.96);
        }
        float3 q = pos;
        
        float scale = 1.0;
        float3 texWeights = abs(nor);
        texWeights = pow(texWeights, 2.0);
        texWeights = saturate(texWeights);
        texWeights.xy = pow(texWeights.xy, 1.5);
        texWeights = normalize(texWeights);
        
        float3x3 mats = float3x3(ground[texIndex].sample(defaultSampler, texScale * q.yz).rgb,
                     ground[texIndex].sample(defaultSampler, texScale * q.xz).rgb,
                     ground[texIndex].sample(defaultSampler, texScale * q.xy).rgb);
        material = scale * mats * texWeights;
        material = pow(material, 2.0);
        if ( m < 2 ) { material = float3(0.8); }
        
        mats = float3x3(ground[texIndex+2].sample(defaultSampler, texScale * q.yz).rgb,
                     ground[texIndex+2].sample(defaultSampler, texScale * q.xz).rgb,
                     ground[texIndex+2].sample(defaultSampler, texScale * q.xy).rgb);
        hNor = scale * mats * texWeights;
        hNor = normalize(hNor);

        float3 mats1x3 = float3(ground[texIndex+3].sample(defaultSampler, texScale * q.yz).r,
                      ground[texIndex+3].sample(defaultSampler, texScale * q.xz).r,
                      ground[texIndex+3].sample(defaultSampler, texScale * q.xy).r);

        rou = scale * dot(texWeights, mats1x3);

        a = pow(rou, 2.0);
    }
//
//    if (m > 0. && m < 35.) {
//        f0 = float3(.015);
//        float3 q = pos;
//        q.z *= 0.325;
//        if (m < 10.) {
//            f0 = float3(.99,0.98,0.97);
//            q = transformPendulumPoint(pos, time);
//        }
//
//        float scale = 1.0;
//        float3 texWeights = abs(nor);
//        texWeights = pow(texWeights, 2.0);
//        texWeights = saturate(texWeights);
//        texWeights.xy = pow(texWeights.xy, 1.5);
//        texWeights = normalize(texWeights);
//
//        float3x3 mats = float3x3(ground[texIndex].sample(defaultSampler, texScale * q.yz).rgb,
//                     ground[texIndex].sample(defaultSampler, texScale * q.xz).rgb,
//                     ground[texIndex].sample(defaultSampler, texScale * q.xy).rgb);
//        material = scale * mats * texWeights;
//        material = pow(material, 2.0);
//
//        mats = float3x3(ground[texIndex+2].sample(defaultSampler, texScale * q.yz).rgb,
//                     ground[texIndex+2].sample(defaultSampler, texScale * q.xz).rgb,
//                     ground[texIndex+2].sample(defaultSampler, texScale * q.xy).rgb);
//        hNor = scale * mats * texWeights;
//        hNor = normalize(hNor);
//
//        float3 mats1x3 = float3(ground[texIndex+3].sample(defaultSampler, texScale * q.yz).r,
//                      ground[texIndex+3].sample(defaultSampler, texScale * q.xz).r,
//                      ground[texIndex+3].sample(defaultSampler, texScale * q.xy).r);
//
//        rou = scale * dot(texWeights, mats1x3);
//        // }
//
////        a = pow(rou, 1.08);
//        a = rou;
//    }
    
    // lighting
    if ( m > 0. && m < 20. ) {
        float3 l = float3(0.0);
        float occ = calcAO(pos, nor, time);
        
        // top - BRDF
        {
            float3 lig = LPOS() * float3(1,-1,1);
            
            float dif = clamp(dot(lig,nor), 0.0, 1.0);
            float difrd = clamp(dot(-rd, nor), 0.0, 1.0);

            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);

            float shadow = calcSoftshadow(pos, lig - float3(0,0,5.0), 0.0021, 5.0, 5.0, time);
//            shadow = 0.0;
            float3 clr = 10.5*pow(LCOL(), 2.0);
            
            // fresnel
            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            
            // distribution
            float a2 = pow(a, 2.0);
            float dTr = a2 / ( 3.14159*pow((pow(dot(hNor, nor), 2.0) * (a2 - 1.0) + 1.0), 2.0) );

            float k = pow((rou + 1.0)/2.0, 2.0) / 8.0;
            float G1L = dif / ( dif * (1.0-k) + k );
            float G1V = difrd / ( difrd * (1.0-k) + k );
            
            float GSmith = G1L * G1V;
            
            // full spectral light addition
            float3 spe = clamp((fSch * dTr * GSmith) / ( 4.0 * dif * difrd ), 0.0, 1.0);
            
            l+= spe * clr * occ * shadow;
            l+= (rou+(1.0-rou)*dif) * clr * occ * shadow * material; // dif
            
            float bounce = abs(dot(-lig, hal));
            l += rou * bounce * 0.1 * clr * material;

        }
        
        {
            float3 lig = normalize(float3(0.0, 0.28, 0.56));
            float dif = clamp(dot(lig,nor), 0.0, 1.0);
            float difrd = clamp(dot(-rd, nor), 0.0, 1.0);
            
            float3 rfl = reflect(rd, nor);

            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);

            float shadow = calcSoftshadow(pos, lig, 0.0015, 5.0, 5.0, time);
//            shadow = 0.0;
   
            float3 clr = float3(.335, .364, .415);
            clr = 10.5 * pow(clr * LCOL(), 2.0);
            
            // fresnel
            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            
            // distribution
            float a2 = pow(a, 2.0);
            float dTr = a2 / ( 3.14159*pow((pow(dot(hNor, nor), 2.0) * (a2 - 1.0) + 1.0), 2.0) );
            
            float k = pow((rou + 1.0)/2.0, 2.0) / 8.0;
            float G1L = dif / ( dif * (1.0-k) + k );
            float G1V = difrd / ( difrd * (1.0-k) + k );
            
            float GSmith = G1L * G1V;

            float3 env = m < 10. ? rou * 1.5 * cubemap.sample(defaultSampler, rfl).bgr : 1.0;
            
            // full spectral light addition
            float3 spe = clamp((fSch * dTr * GSmith) / ( 4.0 * dif * difrd ), 0.0, 1.0);
            
            if ( m < 10. ) {
                l+= spe * clr * (occ * shadow * env + 0.5*env);
                l+= (rou+(1.0-rou)*dif) * clr * occ * shadow * material; // dif
            } else {
                l+= spe * clr * occ * shadow;
                l+= (rou+(1.0-rou)*dif) * clr * occ * shadow * material; // dif
            }
        }
    
        col = ((0.05 * material) + (0.95 * l));
    }
//    
//    // face
//    else if (m > 20 && m < 35.) {
//        float3 l = float3(0.0);
//
//        // march towards clock face
//        float res = -1.0;
//        float tmin = 0.0021;
//        float tmax = 500.0;
//
//        float eps = 0.001;
//        float t = tmin;
// 
//        for( int i = 0; i < 256 && t < tmax; i++) {
//            float3 q = ro + rd*t;
//            float h = mapGlass(q, time);
//
//            if( abs(h) < eps){
//                res = t;
//                break;
//            }
//
//            t += h/2.0;
//        }
//
//        if (res > 0.) {
//            rou = pow((rou/2.0), 2.0);
//            float3 facePosition = ro + rd*res;
//            float3 faceNor = float3(0,0,1);
//            facePosition.y += 15.0;
//            facePosition.x += 20.0;
//            float cx = fmod((facePosition.x) + 2.0, 4.0) - 2.0;
//            float cy = fmod((facePosition.y) + 2.0, 4.0) - 2.0;
//            float2 iq = float2(floor((facePosition.x+2.0)/4.0), floor((facePosition.y+2.0)/4.0));
//            
//            float2 uv = float2(cx,cy);
//            float3 q = float3(uv,pos.z+0.02);
//            
//            float3 material = ground[0].sample(defaultSampler, float2(1,-1) * 1.025 * uv + float2(0.5,0.265)).bgr;
//            
//            // make clock hands
//            float2 center = float2(0,0.75);
//            float2 h = q.xy - center;
//            
//            // center
//            material -= 0.88*float3(smoothstep(0.0285,0.028, length(h)));
//
//            float angle = -(time*0.8);
//            float cs = cos(angle+iq.x*123.5553);
//            float sn = sin(angle+iq.x*123.5553);
//            float2x2 rot = float2x2(cs, -sn, sn, cs);
//            
//            h = rot * h;
//
//            // hour
//            if (h.x > 0) {
//                material -= smoothstep(0.00125, 0.001, abs(h.y)/((1.0-2.0*h.x)*5.)) * smoothstep(0.235,0.234, h.x);
//                if ( 25.0*(h.x+0.09) > 3.141593653/2.0) {
//                    material -= smoothstep(0.251,0.25, h.x) * smoothstep(0.00125, 0.001, abs(h.y) - (2.0*cos(25.0*(h.x+0.09))/(6.5*pow(10.0*(h.x+0.09),3.0))));
//                }
//                material = clamp(material, 0., 1.);
//            }
//            
//            angle = 3.14*cos(cos(0.8*time+0.1*time*iq.x) + cos(0.8*time+2.553*iq.y));
//            cs = cos(angle);
//            sn = sin(angle);
//            rot = float2x2(cs,-sn,sn,cs);
//            
//            h = rot * h;
//            
//            // minute
//            if (h.x > 0) {
//                material -= smoothstep(0.00125, 0.001, abs(h.y)/((1.0-2.0*h.x)*5.)) * smoothstep(0.2851,0.285, h.x);
//                if ( 25.0*(h.x+0.04) > 3.141593653/2.0) {
//                    material -= smoothstep(0.2851,0.285, h.x) * smoothstep(0.00125, 0.001, abs(h.y) - (1.8*cos(25.0*(h.x+0.04))/(6.5*pow(10.0*(h.x+0.04),3.0))));
//                }
//                material = clamp(material, 0., 1.);
//            }
//                                  
//            // add middle knob
//            material += 0.88*float3(0.97,0.94,0.93) * smoothstep(0.0135,0.013, length(h));
//            material = pow(material, 1/.4545);
//          
//            float occ = calcAO(q, faceNor, time);
//            float3 lig = normalize(float3(0.0, 0.28, 0.56));
//
//            float dif = clamp(dot(lig,faceNor), 0.0, 1.0);
//            float difrd = clamp(dot(-rd, faceNor), 0.0, 1.0);
//
//            float3 hal  = normalize(lig - rd);
//            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
//            
//            float shadow = smoothstep(0.59, 0.415, length(float3(0,0.75,0) - q));
//            shadow = clamp(shadow * shadow * (3.0 - 2.0*shadow),0.,1.);
//            
//            float3 clr = float3(.335, .364, .415);
//            clr = 8.5*pow(clr * LCOL(), 2.0);
//
//            float3 rfl = reflect( rd, nor );
//            float3 env = cubemap.sample(defaultSampler, rfl.zyx * float3(-1,1,1)).bgr;
//            env = pow(env, 2.0);
////            f0 = env;
//            
//            // fresnel
//            float3 fSch = f0 + (float3(1.0) - f0)*pow(fre, 5.0);
//
//            // distribution
//            float a2 = pow(a, 2.0);
//            float dTr = a2 / ( 3.14159*pow((pow(dot(hNor, faceNor), 2.0) * (a2 - 1.0) + 1.0), 2.0) );
//
//            float k = pow((rou + 1.0), 2.0) / 8.0;
//            float G1L = dif / ( dif * (1.0-k) + k );
//            float G1V = difrd / ( difrd * (1.0-k) + k );
//
//            float GSmith = G1L * G1V;
//
//            // full spectral light addition
//            float3 spe = clamp((fSch * dTr * GSmith) / ( 4.0 * dif * difrd ), 0.0, 1.0);
//            
////            l+= spe * shadow * (1.5 * clr * material + pow(1.0+dot(nor,rd),1.0)*2.5*env);
////            l+= (rou + (1.0-rou)*dif) * 1.5 * shadow * clr * occ * material;
//            
//            float3 rflEnv = env * (1.05 - pow(saturate(dot(hal, nor)), 8.0));
//            l += spe * clr * material * shadow * occ;
//            l += (rou + (1.0-rou)*dif) * shadow * material * occ;
//            l += clr * rflEnv * 0.8 * shadow;
////            l = pow(1.0 + dot(nor,-rd),5.0);
//            col = l;
//        } else {
//            col = float3(0.0);
//        }
//    }
    // water - shade separately
    else if ( m > 35. && m < 50.)
    {
        material = 0.5*float3(.035, .064, .415);
        material = pow(material, 2.0);
        nor = waterBumpMap(pos, time);
        float3 lig = LPOS();
        
        float ndotl = dot(nor, lig);
        float ndotr = dot(nor, rd);
        float fresnel = pow(1.0-abs(ndotr),5.);
        
        float3 skyReflect = skyColor(pos, reflect(rd, nor));
        skyReflect = skyReflect * skyReflect * (3.0 - 2.0 * skyReflect);

        col = material + fresnel * skyReflect;
        
        col = col * LCOL() * (0.5 + 0.5 * ndotl);
    }
    
    return saturate( float4(col, t/2000.0) ); // / tmax in raycast
    
}

float2 csqr( float2 a )  { return float2( a.x*a.x - a.y*a.y, 2.*a.x*a.y  ); }

kernel void compute(texture2d<float, access::write> output [[texture(0)]],
                    texturecube<float> cubemap [[texture(1)]],
                    array<texture2d<float, access::sample>, 20> ground [[texture(2)]],
                    sampler defaultSampler [[sampler(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    constant Uniforms &uniforms [[buffer(0)]])
{
    float change = 0.0; // 20 was good
    float t = uniforms.time + 30.0;
    float3 ro = float3( 0.0, change+4.2, 10.25);
    float3 ta = float3( 0.0, change+4.09, 0.5);
    float3x3 ca = setCamera(ro, ta, 0.0);

    int width = output.get_width();
    int height = output.get_height();
    float aspect = uniforms.resolution[0] / uniforms.resolution[1];
    float2 uv = float2(gid) / float2(width, height);
    
    uv = uv * 2.0 - 1.0;
    //uv.x *= aspect;
    uv.y *= -1.0;
    
    // jitter subpixel position each frame
    float2 o = (hash2(uv+t*0.1) / uniforms.resolution);
    float2 p = float2(aspect, 1.0) * (uv);
    
    float3 rd = ca * normalize( float3(p, 2.2) );
    
    float2 px =  (uv+float2(1.0,0.0) - float2(0.5));
    float2 py =  (uv+float2(0.0,1.0) - float2(0.5));
    float3 rdx = ca * normalize( float3(px, 2.5));
    float3 rdy = ca * normalize( float3(py, 2.5));

    float4 color = render( ro, rd, rdx, rdy, t, ground, cubemap, defaultSampler );

    color.rgb = pow(color.rgb, float3(0.4545));

    color.rgb = min(color.rgb, 1.0);

    // dithering
    //total += (1.0/100.0) * sin(uv.x*944.111)*sin(uv.y*843.32);
    output.write(color, gid);
}
