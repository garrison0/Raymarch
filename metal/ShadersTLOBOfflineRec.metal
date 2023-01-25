#include <metal_stdlib>
#include "ShaderTypes.h"
using namespace metal;

float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
float4 mod289(float4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
float4 perm(float4 x){return mod289(((x * 34.0) + 1.0) * x);}

// https://arxiv.org/pdf/2004.06278v3.pdf
uint32_t squares(uint64_t ctr, uint64_t key) {
   uint64_t x, y, z;

   y = x = ctr * key; z = y + key;

   x = x*x + y; x = (x>>32) | (x<<32);       /* round 1 */

   x = x*x + z; x = (x>>32) | (x<<32);       /* round 2 */

   return (x*x + y) >> 32;                   /* round 3 */
}

float hash(float3 p) {
    p = fract(p * 0.13);
    p += dot(p, p.yzx + 3.333);
    return fract(sin(dot(p,float3(-243.3423,23,-23232.2)))*223.1122*(p.x + p.y) * p.z);
}

//float hash (float3 n)
//{
//    return fract(sin(dot(n, 0.4*float3(523.456789, 734.112222, 587.654321))) * 54321.9876 );
//}

//float random(float3 p) {
//    uint64_t ctr = (uint64_t) ( __UINT32_MAX__ * hash(p) );
//    uint64_t key = 0xa31d47fcba69cf13;
//    uint32_t random = squares(ctr, key);
//    return random * 0.0000000003;
//}

float noise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float3 step = 3.0*float3(110,241,171);
    
    float3 u = f * f * (3.0 - 2.0 * f);
    float n = dot(i, step);
    
    return mix(mix(mix( hash(n + dot(step, float3(0, 0, 0))), hash(n + dot(step, float3(1, 0, 0))), u.x),
                       mix( hash(n + dot(step, float3(0, 1, 0))), hash(n + dot(step, float3(1, 1, 0))), u.x), u.y),
                   mix(mix( hash(n + dot(step, float3(0, 0, 1))), hash(n + dot(step, float3(1, 0, 1))), u.x),
                       mix( hash(n + dot(step, float3(0, 1, 1))), hash(n + dot(step, float3(1, 1, 1))), u.x), u.y), u.z);
}

// set to 8 for details
float fbm( float3 p, int octaves ) {
//    p *= 20.;
    const float3x3 m3 = float3x3( 0.00,  0.80,  0.60,
                                 -0.80,  0.36, -0.48,
                                 -0.60, -0.48,  0.64 );
    float val = 0.0;
    float amp = 0.5;
    for(int i = 0; i < octaves; i++) {
        val += amp * noise(p); p = m3 * p * (2.0 + 0.1*i);
        amp *= 0.5;
    }
    return val;
}

float4 permute(float4 x){return fmod(((x*34.0)+1.0)*x, 289.0);}
float4 taylorInvSqrt(float4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(float3 v){
  const float2  C = float2(1.0/6.0, 1.0/3.0) ;
  const float4  D = float4(0.0, 0.5, 1.0, 2.0);

// First corner
  float3 i  = floor(v + dot(v, C.yyy) );
  float3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  float3 g = step(x0.yzx, x0.xyz);
  float3 l = 1.0 - g;
  float3 i1 = min( g.xyz, l.zxy );
  float3 i2 = max( g.xyz, l.zxy );

  //  x0 = x0 - 0. + 0.0 * C
  float3 x1 = x0 - i1 + 1.0 * C.xxx;
  float3 x2 = x0 - i2 + 2.0 * C.xxx;
  float3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
  i = fmod(i, 289.0 );
  float4 p = permute( permute( permute(
             i.z + float4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + float4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + float4(0.0, i1.x, i2.x, 1.0 ));

// Gradients
// ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0/7.0; // N=7
  float3  ns = n_ * D.wyz - D.xzx;

  float4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

  float4 x_ = floor(j * ns.z);
  float4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  float4 x = x_ *ns.x + ns.yyyy;
  float4 y = y_ *ns.x + ns.yyyy;
  float4 h = 1.0 - abs(x) - abs(y);

  float4 b0 = float4( x.xy, y.xy );
  float4 b1 = float4( x.zw, y.zw );

  float4 s0 = floor(b0)*2.0 + 1.0;
  float4 s1 = floor(b1)*2.0 + 1.0;
  float4 sh = -step(h, float4(0.0));

  float4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  float4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  float3 p0 = float3(a0.xy,h.x);
  float3 p1 = float3(a0.zw,h.y);
  float3 p2 = float3(a1.xy,h.z);
  float3 p3 = float3(a1.zw,h.w);

//Normalise gradients
  float4 norm = taylorInvSqrt(float4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  float4 m = max(0.6 - float4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, float4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}


float2 opU(float2 d1, float2 d2) {
    return d1.x < d2.x ? d1 : d2;
}

float2 opSmoothU( float2 d1, float2 d2, float k)
{
    float colorSmoothness = k * 4.0;
    float interpo = clamp( 0.5 + 0.5 * (d1.x - d2.x) / colorSmoothness, 0.0, 1.0 );
    float h = max( k - abs(d1.x - d2.x), 0.0) / k;
    float diff = h*h*h*k*(1.0/6.0);
    return float2( min(d1.x, d2.x) - diff,
                 mix(d1.y, d2.y, interpo) - k * interpo * ( interpo - 1.0) );
}

float opIntersection( float d1, float d2 ) { return max(d1,d2); }

float sdfSphere(float3 p, float r) {
    return length( p ) - r;
}

float2 sdCapsule( float3 p, float3 a, float3 b )
{
  float3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return float2( length( pa - ba*h ) - 0.01, h );
}

float sdBox( float3 p, float3 b )
{
    float3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float2 mapFlux(float3 p, float time, float3 pos) {
    float2 res = float2(1e10, 0.0);

    if (p.y < -0.675) { return res; }
    
    float3 p3 = p;
    float ss = sdBox(p + pos, float3(0.8, 1.25, 0.5));
    float bounds = (smoothstep(0.25, -0.75, ss)) * smoothstep(-0.68, -0.675, p.y) * smoothstep(1.75, -0.15, p.y);
    
    {
        float x = time;
        
        float sTime = time * 1.0;
        float modTime = fmod(sTime, 12.0);
        float width = 4.0;
        
        float fadeIn = min(time*time/18000.0, 1.0);
        fadeIn = fadeIn * (0.5 + 0.5 * sin(( 4.0 * 3.141593653 / 2.0 ) + 3.141593653 * time / 60.0));
//        fadeIn = 1.0;
        float scale1 = 0.11 + 0.89*(1 / (1 + pow((float)modTime - width, 4.0)));
        float scale2 = 0.12 + 0.88*(1 / (1 + pow((float)modTime - 2.0*width, 4.0)));
        
        float sig4 = fadeIn * (0.5 + 0.5*cos(0.35*x));
        float sig = fadeIn * (scale2 * (0.5 + 0.5*sin(0.35*x)));
        float sig2 = fadeIn * (0.5 + 0.5*sin(0.35*x + 4.0*3.141592653/3.0));
        float sig3 = fadeIn * (scale1 * (0.5 + 0.5*sin(0.35*x + 2.0*3.141592653/3.0)));
        float x2 = x / 20.0;
        float sig5 = fadeIn * scale2 * (0.5 + 0.5 * sin(0.22*x2 + sin(1.12132*x2 + cos(0.72453*x2))));
        
        float NoiseScale = 1.093;//0.25*sig2 + 1.09391;
        float NoiseIsoline = 0.73319 * bounds;

        p3 = p3 - time * float3(0,0.05,0);
        p3 = p3 / NoiseScale;

        float3 moveDown = x*float3(-0.03,0.88,-0.2);
        x = 1/x;
        float noise = NoiseScale * (fbm(0.74 * p3
                                      + 0.24 * sig3 * fbm( p3*x + sig5*0.1*sin(1.0*p3*x*sig2+0.1*sin(x*sig4*p3+0.1*moveDown*sin(10.0*sig3*x*p3))) - 0.1*moveDown, 5.0)
                                      + 0.72 * sig * snoise(p3 - 0.2*moveDown + sig2*0.6*snoise(1.8*p3-0.55*moveDown)),
                                    5.0) - NoiseIsoline); // used to be 0.76 * ...

        res = opU(res, float2(noise, 10.0));
    }

    return res;
}

float2 map(float3 p, float time)  {
    float2 res = float2(1e10, 0.0);

//    return float2(sdfSphere(p + float3(0,0,1), 0.5), 15.0);
    float dToBound = sdBox(p + float3(0.0, 0.0, 0.75), float3(1.05, 1.35, 1.65));
    if (dToBound > 0.001) {
        res = opU(res, float2(dToBound, 10.0));
    } else {
        float2 dFlux = mapFlux(p, time, float3(0,0,1));
        res = opU(res, dFlux);
    }
    
    return res;
}

float calcSoftshadow( float3 ro, float3 rd, float mint, float maxt, float k, float time )
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t<maxt; )
    {
        float h = map(ro + rd*t, time).x;
        if( h<0.000001 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h/5.0;
    }
    return res;
}

float calcAO( float3 pos, float3 nor, float time)
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

float3 calcNormal( float3 p, float time )
{
    const float eps = 0.0001;
    const float2 h = float2(eps,0);
    return normalize( float3(map(p+h.xyy, time).x - map(p-h.xyy, time).x,
                             map(p+h.yxy, time).x - map(p-h.yxy, time).x,
                             map(p+h.yyx, time).x - map(p-h.yyx, time).x ) );
}

float2 raycast(float3 ro, float3 rd, float time){
    float2 res = float2(-1.0,-1.0);

    float tmin = 0.002;
    float tmax = 100.0;
    
    float eps = 0.001;
    float t = tmin;
    for( int i = 0; i < 1250 && t < tmax; i++) {
        float2 h = map( ro + rd*t, time );

        if( abs(h.x) < eps){
            res = float2(t, h.y);
            break;
        }

        t += h.x/12.0;
    }

    return res;
}

float3x3 setCamera(float3 ro, float3 ta, float cr )
{
    float3 cw = normalize( ta - ro );
    float3 cp = float3( sin(cr), cos(cr), 0.0 );
    float3 cu = normalize( cross(cw,cp) );
    float3 cv =          ( cross(cu,cw) );
    return float3x3(cu,cv,cw);
}

float3 render(float3 ro, float3 rd, float3 rdx, float3 rdy, float time, array<texture2d<float, access::sample>, 20> ground, sampler defaultSampler)
{
    float2 res = raycast(ro,rd, time);
    float t = res.x;
    float m = res.y;

    float3 col = float3(0.8);
    
    if (m > 0.0) {
        float3 lin = float3(0);
        float3 mat = float3(0);
        
        float3 p = ro + rd * t;
        float3 normal = calcNormal(p, time);
        
//        float3 macroNormal = normal;
        float2 K = float2(0,1); // amount, power for specular
        float3 f0 = float3(0.035);
        float rou = 1.0; // roughness
        float3 hNor = float3(0); // microfacet normal - is getting it from texture better than 'hal'?
        float a = 1.0; // blinn-phong shininess / roughness
        float occ = 1.0;
        float3 pMoved = p;
        float difScalar = 0.0;
        
        if (m < 10.0)
        {
            mat = float3(0.15, 0.1, 0.2);
        }
        else if (m < 20.0) {
            float scalars[6];
            float Ks[5] = {0.35,0.35,0.1,0.3,0.16};
            float difScale[5] = {8, 20, 40, 8, 20};
            float3 f0s[5];
            
            // snow, asphalt, grass, sand, leaves
            f0s[0] = float3(0.02);
            f0s[1] = float3(0.03);
            f0s[2] = float3(0.03);
            f0s[3] = float3(0.05);
            f0s[4] = float3(0.03);
            
            for (int i = 0; i < 6; i++)
            { // render time*60 in 74 sec == roughly full thing sped up
                float sTime = time * 0.767 / 60.0; // realtime/(5*1.14) = 74 min, 14.8 min each
                float modTime = fmod(sTime, 13.0);
                float width = 2.6;
                scalars[i] = 1 / (1 + pow((float)modTime - i*width, 4.0));
                scalars[i] = max(scalars[i], 1 / (1 + pow((float)sTime - i*width, 4.0)));
            }
            
            for (int i = 0; i < 5; i++) {
                int texIndex = i*4;
                float scale = i == 0 ? scalars[0] + scalars[5] : scalars[i];
                
                pMoved = p - time * float3(0,0.05,0);
                pMoved *= 1.1;
                float3 texWeights = abs(normal) - 0.2;
                texWeights = pow(texWeights, 2.0);
                texWeights = saturate(texWeights);
                texWeights.xy = pow(texWeights.xy, 1.5);
                texWeights = normalize(texWeights);
                                       
                float3x3 mats = float3x3(ground[texIndex].sample(defaultSampler, pMoved.yz).rgb,
                             ground[texIndex].sample(defaultSampler, pMoved.xz).rgb,
                             ground[texIndex].sample(defaultSampler, pMoved.xy).rgb);
                mat += scale * mats * texWeights;
                
                mats = float3x3(ground[texIndex+2].sample(defaultSampler, pMoved.yz).rgb,
                             ground[texIndex+2].sample(defaultSampler, pMoved.xz).rgb,
                             ground[texIndex+2].sample(defaultSampler, pMoved.xy).rgb);
                hNor += scale * mats * texWeights;
                
                float3 mats1x3 = float3(ground[texIndex+3].sample(defaultSampler, pMoved.yz).r,
                              ground[texIndex+3].sample(defaultSampler, pMoved.xz).r,
                              ground[texIndex+3].sample(defaultSampler, pMoved.xy).r);
                rou += scale * dot(texWeights, mats1x3);

                a += scale * 16.0 * (0.07 + 0.93*(1.0 - rou));
                
                K += scale * float2(Ks[i], 16.0);
                f0 += scale * f0s[i];
                difScalar += scale * difScale[i];
            }
        }
        
        normal = normalize(normal + hNor);
        occ = calcAO(p, normal, time);
        
        // top light BRDF
        {
            float3 lig = normalize(float3(-0.123, -1.55, 0.44));
            float3 clr = 1.9*float3(1.24,1.17,1.24);
            float dif = dot(lig, normal);
            dif *= occ;
            
            float shadow = calcSoftshadow(p, lig, 0.0021, 20.0, 4, time);
//            float shadow = 1.0;
            // fresnel
            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            
            // geometrySmith;
            // dBlinnPhong;
            float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(normal, hal), 0.0, 1.0), a);
            
            float m = rou * sqrt(2 / 3.141593653);
            float g1SchlickVH = dot(normal, ro) / ( dot(normal, ro) * (1 - m) + m);
            float g1SchlickLH = dot(normal, lig) / ( dot(normal, lig) * (1 - m) + m);
//
            float gSmith = g1SchlickLH * g1SchlickVH;
            
            float3 spe = gSmith * dBlinnPhong * fSch / ( 4.0 * dot(normal, lig) * dot(normal, ro));
            
            float3 cshadow = pow( float3(shadow), float3(1.3, 1.5, 1.8) );

            float sky = clamp( 0.5 - 0.5*normal.y, 0.0, 1.0 );
            float ind = clamp( dot( normal, normalize(lig*float3(-1.0)) ), 0.0, 1.0 );

            lin += K.x * clr * 6.0 * spe * cshadow; // spec
            lin += (1 - K.x) * difScalar * dif * cshadow * mat * mat; // dif - 8
            lin += 1.5*sky*float3(0.16,0.20,0.35)*occ*mat; // sky
            lin += 1.25*ind*float3(0.40,0.28,0.20)*occ*mat; // bounce
        }
        
        {
            float3 lig = float3(0.0, -1.0, -3.4);
            float3 clr = 1.4*float3(1.34,1.27,0.99);
            float3 hal  = normalize(lig - rd);
            float dif = clamp(dot(normal, lig), 0.0, 1.0);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
            float3 fSch = clamp(f0 + (float3(1) - f0)*pow(fre, 5.0), 0,1);
            lin += 0.4*fSch*fre* occ*mat*clr*dif * mat;
            lin += 0.3*fSch * clr*dif * mat;
        }
        
        float c = smoothstep(0.92, -0.15, p.y);
        col = (1.0-c) * col + c * ((0.05 * mat) + (0.95 * lin));
    }
    
    return col;
}

kernel void compute(texture2d<float, access::write> output [[texture(0)]],
                    array<texture2d<float, access::sample>, 20> ground [[texture(1)]],
                    sampler defaultSampler [[sampler(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    constant Uniforms &uniforms [[buffer(0)]])
{
    float3 ro = float3( 0.0, -0.7, 1.5 );
    float3 ta = float3( 0.0, -0.45, 0.5 );
    float3x3 ca = setCamera(ro, ta, 0.0);
    float3 total = float3(0.0);
    
    int width = output.get_width();
    int height = output.get_height();
    float aspect = uniforms.resolution[0] / uniforms.resolution[1];
    float2 uv = float2(gid) / float2(width, height);
    
//    uv = uv * 2.0 - 1.0;
//    uv.x *= aspect;

    int AA = 2;
    int m = 0;
    for (int n=0; n < AA; n++) {
//        for(int m = 0; m < AA && m <= n; m++) {
            float2 o = (float2(float(m), float(n)) / uniforms.resolution) / float(AA);
            float2 p = float2(aspect, 1.0) * ( 1.75*(uv+o) - float2(0.875));
            float time = uniforms.time + 1.0 * (1.0/40.0) * float(m * n * AA) / float(AA*AA);
            
            time -= 6 * 60;
//            time -= 10 * 60;
            // manually draw a 'source'
            float linePos = -0.452;
            float lineLen = 0.7;
            float lineWidth = 0.003;
            if (p.x > -lineLen && p.x < lineLen && p.y < linePos && p.y > linePos - lineWidth) {
                total += float3(0.1);
            }
            else {
                float3 rd = ca * normalize( float3(p, 2.2) );
                
                // ray differentials
                float2 px =  (uv+float2(1.0,0.0) - float2(0.5));
                float2 py =  (uv+float2(0.0,1.0) - float2(0.5));
                float3 rdx = ca * normalize( float3(px, 2.5));
                float3 rdy = ca * normalize( float3(py, 2.5));
                
                float3 color = render( ro, rd, rdx, rdy, 0.2*time, ground, defaultSampler );

                color = pow(color, float3(0.4545));

                total += color;
                
//            }
        }
    }
    
    
    total /= float(AA);
    
    output.write(float4(total, 1.0), gid);
}
