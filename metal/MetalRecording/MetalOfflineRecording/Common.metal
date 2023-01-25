#include <metal_stdlib>

using namespace metal;
#include "Common.h"

// scene specific globals
// light (sun) position, color
float3 LPOS() {
    return -normalize(float3(0.0, 0.5, 0.13));
}

float3 LCOL() {
    return float3(0.95, 0.904, 0.94);
}

// noise, utilities, etc.
float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
float4 mod289(float4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
float4 perm(float4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noise(float3 p){
    float3 a = floor(p);
    float3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    float4 b = a.xxyy + float4(0.0, 1.0, 0.0, 1.0);
    float4 k1 = perm(b.xyxy);
    float4 k2 = perm(k1.xyxy + b.zzww);

    float4 c = k2 + a.zzzz;
    float4 k3 = perm(c);
    float4 k4 = perm(c + 1.0);

    float4 o1 = fract(k3 * (1.0 / 41.0));
    float4 o2 = fract(k4 * (1.0 / 41.0));

    float4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    float2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

float fbm( float3 p ) {
    float f = 0.0;
    const float3x3 m3 = float3x3( 0.00,  0.80,  0.60,
                         -0.80,  0.36, -0.48,
                         -0.60, -0.48,  0.64 );

    f += 0.5000*noise( p ); p = m3*p*1.62;
    f += 0.2500*noise( p ); p = m3*p*1.93;
    f += 0.1250*noise( p ); p = m3*p*1.7;
    f += 0.0625*noise( p ); p = m3*p*1.623;
    f += 0.0625*noise( p );
    return f/0.999;
}

float3 hash3( float2 p ){
    float3 q = float3( dot(p,float2(127.1,311.7)),
                   dot(p,float2(269.5,183.3)),
                   dot(p,float2(419.2,371.9)) );
    return fract(sin(q)*43758.5453);
}

float2 hash2( float2 p ){
    float2 q = float2( dot(p,float2(127.1,311.7)),
                   dot(p,float2(269.5,183.3)));
    return fract(sin(q)*43758.5453);
}

float vnoise( float2 x, float u, float v )
{
    float2 p = floor(x);
    float2 f = fract(x);

    float k = 1.0 + 63.0*pow(1.0-v,4.0);
    float va = 0.0;
    float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        float2  g = float2( float(i), float(j) );
        float3  o = hash3( p + g )*float3(u,u,1.0);
        float2  r = g - f + o.xy;
        float d = dot(r,r);
        float w = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
        va += w*o.z;
        wt += w;
    }

    return va/wt;
}

float vnoiseOctaves (float2 x, float u, float v ) {
    float t = -.5;

    for (float f = 1.0 ; f <= 3.0 ; f++ ){
        float power = pow( 2.0, f );
        t += abs( vnoise( power * x, u, v ) / power );
    }

    return t;
}

float3 mod289(float3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float4 permute(float4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

float4 taylorInvSqrt(float4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(float3 v)
{
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

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    float3 x1 = x0 - i1 + C.xxx;
    float3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    float3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i);
    float4 p = permute( permute( permute(
                i.z + float4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + float4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + float4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    float3  ns = n_ * D.wyz - D.xzx;

    float4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    float4 x_ = floor(j * ns.z);
    float4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    float4 x = x_ *ns.x + ns.yyyy;
    float4 y = y_ *ns.x + ns.yyyy;
    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4( x.xy, y.xy );
    float4 b1 = float4( x.zw, y.zw );

    //float4 s0 = float4(lessThan(b0,0.0))*2.0 - 1.0;
    //float4 s1 = float4(lessThan(b1,0.0))*2.0 - 1.0;
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
    float4 m = max(0.5 - float4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 105.0 * dot( m*m, float4( dot(p0,x0), dot(p1,x1),
                                    dot(p2,x2), dot(p3,x3) ) );
}

float curlNoise( float3 p ){

  const float e = 0.1;

  float  n1 = snoise(float3(p.x, p.y + e, p.z));
  float  n2 = snoise(float3(p.x, p.y - e, p.z));
  float  n3 = snoise(float3(p.x, p.y, p.z + e));
  float  n4 = snoise(float3(p.x, p.y, p.z - e));
  float  n5 = snoise(float3(p.x + e, p.y, p.z));
  float  n6 = snoise(float3(p.x - e, p.y, p.z));

  float x = n2 - n1 - n4 + n3;
  float y = n4 - n3 - n6 + n5;
  float z = n6 - n5 - n2 + n1;

  const float divisor = 1.0 / ( 2.0 * e );
//  return normalize( float3( x , y , z ) * divisor );
    return length(float3(x,y,z)*divisor);
}

float3 N13(float p) {
   float3 p3 = fract(float3(p) * float3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(float3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

float linearstep( float s, float e, float v ) {
    return clamp( (v-s)*(1./(e-s)), 0., 1. );
}

float3 rotatePoint(float3 p, float3 n, float theta) {
    float4 q = float4(cos(theta / 2.0), sin(theta / 2.0) * n);
    float3 temp = cross(q.xyz, p) + q.w * p;
    float3 rotated = p + 2.0*cross(q.xyz, temp);
    return rotated;
}

float Saw(float b, float t) {
    return smoothstep(0., b, t)*smoothstep(1., b, t);
}

// Triplanar mapping
float tex3D(sampler defaultSampler, texture2d<float, access::sample> tex, float3 pos, float3 nor, float texScale)
{
    float3 texWeights = abs(nor);
    texWeights = pow(texWeights, 2.0);
    texWeights = saturate(texWeights);
    texWeights.xy = pow(texWeights.xy, 1.5);
    texWeights = normalize(texWeights);
    
    float3 mats = float3(tex.sample(defaultSampler, texScale * pos.yz).r,
                 tex.sample(defaultSampler, texScale * pos.xz).r,
                 tex.sample(defaultSampler, texScale * pos.xy).r);
    return dot(mats, texWeights);
}

// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total.
float3 doBumpMap( sampler defaultSampler, texture2d<float, access::sample> tex, float texScale, float3 p, float3 nor, float bumpfactor){
   
    const float eps = 0.001;
    float3 grad = float3( tex3D(defaultSampler, tex, float3(p.x-eps, p.y, p.z), nor, texScale),
                        tex3D(defaultSampler, tex, float3(p.x, p.y-eps, p.z), nor, texScale),
                        tex3D(defaultSampler, tex, float3(p.x, p.y, p.z-eps), nor, texScale));
    
    grad = (grad - tex3D(defaultSampler, tex, p , nor, texScale))/eps;
            
    grad -= nor*dot(nor, grad);
                      
    return normalize( nor + grad*bumpfactor );
}

// procedural coloring trick
float3 palette( float t, float3 a, float3 b, float3 c, float3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

float3x3 setCamera( float3 ro, float3 ta, float cr )
{
    float3 cw = normalize(ta-ro);
    float3 cp = float3(sin(cr), cos(cr),0.0);
    float3 cu = normalize( cross(cw,cp) );
    float3 cv =          ( cross(cu,cw) );
    return float3x3( cu, cv, cw );
}

