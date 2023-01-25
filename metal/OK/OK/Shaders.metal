#include <metal_stdlib>
#include "Shaders.h"
using namespace metal;

#define program 4

float signalOne(float time) {
    float length = 60.0;
    float a = fmod(time, length);
    float ap = 30.0*(0.5 - 0.5*cos( ( 2.0*3.141593653 / length ) * a ));
    return ap;
}

float signalTwo(float time) {
    float length = 60.0;
    float a = fmod(time, length);
    float ap = 30.0 * (0.5 + 0.5*cos( ( 2.0*3.141593653 / length ) * a + 3.141592653/4.0 ));
    return ap;
}

float sizeSignal(float time) {
    float length = 60.0;
    float a = fmod(time, length);
    float ap = 0.5 + 0.5 * sin( ( 2.0*3.141593653 / length ) * a );
    return ap;
}

float Saw(float b, float t) {
    return smoothstep(0., b, t)*smoothstep(1., b, t);
}

float3 LPOS() {
    return 20*normalize(float3(0.0,0.5,-0.6));
}

/*
 
                                                    
                                                    
 `7MN.   `7MF' .g8""8q. `7MMF' .M"""bgd `7MM"""YMM
   MMN.    M .dP'    `YM. MM  ,MI    "Y   MM    `7
   M YMb   M dM'      `MM MM  `MMb.       MM   d
   M  `MN. M MM        MM MM    `YMMNq.   MMmmMM
   M   `MM.M MM.      ,MP MM  .     `MM   MM   Y  ,
   M     YMM `Mb.    ,dP' MM  Mb     dM   MM     ,M
 .JML.    YM   `"bmmd"' .JMML.P"Ybmmd"  .JMMmmmmMMM
                                                    
                                                    
 
*/
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

//// 3d SIMPLEX NOISE /////
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

    for (float f = 1.0 ; f <= 10.0 ; f++ ){
        float power = pow( 2.0, f );
        t += abs( vnoise( power * x, u, v ) / power );
    }

    return t;
}

float3 N13(float p) {
   float3 p3 = fract(float3(p) * float3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(float3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

/*
  
                                       
                                       
  .M"""bgd `7MM"""Yb. `7MM"""YMM
 ,MI    "Y   MM    `Yb. MM    `7
 `MMb.       MM     `Mb MM   d ,pP"Ybd
   `YMMNq.   MM      MM MM""MM 8I   `"
 .     `MM   MM     ,MP MM   Y `YMMMa.
 Mb     dM   MM    ,dP' MM     L.   I8
 P"Ybmmd"  .JMMmmmdP' .JMML.   M9mmmP'
                                       
                                       
 
*/

float2 opU( float2 d1, float2 d2 )
{
    return (d1.x<d2.x) ? d1 : d2;
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

float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }

float opIntersection( float d1, float d2 ) { return max(d1,d2); }

float sdfSphere(float3 p, float r) {
    return length( p ) - r;
}

float sdTorus( float3 p, float2 t )
{
    float2 q = float2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float sdBox( float3 p, float3 b )
{
  float3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdRoundBox( float3 p, float3 b, float r )
{
    float3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdHexPrism( float3 p, float2 h )
{
  const float3 k = float3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
  float2 d = float2(
       length(p.xy-float2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdbEllipsoid( float3 p, float3 r )
{
    float k1 = length(p/r);
    float k2 = length(p/(r*r));
    return k1*(k1-1.0)/k2;
}

float2 sdCapsule( float3 p, float3 a, float3 b )
{
  float3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return float2( length( pa - ba*h ) - 0.01, h );
}

float sdCappedCone(float3 p, float3 a, float3 b, float ra, float rb)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(p-a,p-a);
    float paba = dot(p-a,b-a)/baba;
    float x = sqrt( papa - paba*paba*baba );
    float cax = max(0.0,x-((paba<0.5)?ra:rb));
    float cay = abs(paba-0.5)-0.5;
    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );
    float cbx = x-ra - f*rba;
    float cby = paba - f;
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

float sdCappedCylinderVertical( float3 p, float h, float r )
{
  float2 d = abs(float2(length(p.xz),p.y)) - float2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCappedCylinder(float3 p, float3 a, float3 b, float r)
{
  float3  ba = b - a;
  float3  pa = p - a;
  float baba = dot(ba,ba);
  float paba = dot(pa,ba);
  float x = length(pa*baba-ba*paba) - r*baba;
  float y = abs(paba-baba*0.5)-baba*0.5;
  float x2 = x*x;
  float y2 = y*y*baba;
  float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
  return sign(d)*sqrt(abs(d))/baba;
}

float sdTriangleIsosceles( float2 p, float2 q )
{
    p.x = abs(p.x);
    float2 a = p - q*clamp( dot(p,q)/dot(q,q), 0.0, 1.0 );
    float2 b = p - q*float2( clamp( p.x/q.x, 0.0, 1.0 ), 1.0 );
    float s = -sign( q.y );
    float2 d = min( float2( dot(a,a), s*(p.x*q.y-p.y*q.x) ),
                  float2( dot(b,b), s*(p.y-q.y)  ));
    return -sqrt(d.x)*sign(d.y);
}

float sdExtrudedTriangle(float3 p, float2 q, float h)
{
//    p = float3(p.y, -p.x, p.z);
    float d = sdTriangleIsosceles(p.xy, q);
    float2 w = float2( d, abs(p.z) - h );
    return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}

float sdOctogon( float2 p, float r )
{
    const float3 k = float3(-0.9238795325, 0.3826834323, 0.4142135623 );
    p = abs(p);
    p -= 2.0*min(dot(float2( k.x,k.y),p),0.0)*float2( k.x,k.y);
    p -= 2.0*min(dot(float2(-k.x,k.y),p),0.0)*float2(-k.x,k.y);
    p -= float2(clamp(p.x, -k.z*r, k.z*r), r);
    return length(p)*sign(p.y);
}

float sdExtrudedOctogon(float3 p, float r, float h) {
    float d = sdOctogon(p.xy, r);
    float2 w = float2( d, abs(p.z) - h );
    return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}

float sdCappedTorus( float3 p, float2 sc, float ra, float rb)
{
    p.x = abs(p.x);
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}


/*
 
                                                                                      
                                                                                      
 `7MMM.     ,MMF'      db      `7MM"""Mq.`7MM"""Mq.`7MMF'`7MN.   `7MF' .g8"""bgd
   MMMb    dPMM       ;MM:       MM   `MM. MM   `MM. MM    MMN.    M .dP'     `M
   M YM   ,M MM      ,V^MM.      MM   ,M9  MM   ,M9  MM    M YMb   M dM'       `
   M  Mb  M' MM     ,M  `MM      MMmmdM9   MMmmdM9   MM    M  `MN. M MM
   M  YM.P'  MM     AbmmmqMA     MM        MM        MM    M   `MM.M MM.    `7MMF'
   M  `YM'   MM    A'     VML    MM        MM        MM    M     YMM `Mb.     MM
 .JML. `'  .JMML..AMA.   .AMMA..JMML.    .JMML.    .JMML..JML.    YM   `"bmmmdPY
                                                                                      
                                                                                      
 
*/

float3 rotatePoint(float3 p, float3 n, float theta) {
    float4 q = float4(cos(theta / 2.0), sin(theta / 2.0) * n);
    float3 temp = cross(q.xyz, p) + q.w * p;
    float3 rotated = p + 2.0*cross(q.xyz, temp);
    return rotated;
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
    res = opU(res, float2(sdBox(p + float3(0,4,0), float3(100.0, 1.0, 100.0)), 1.0));

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


// VOLUMETRIC:

float3 evaluateLight( float3 pos )
{
    float3 lightPos = LPOS();
    float3 lightCol = 500.*float3(0.99);
    float3 L = lightPos-pos;
    float lamp = smoothstep(0.9, 1.0, (dot(float3(0,-1.0,0), normalize(pos - lightPos))));
    return lightCol * lamp * 1.0/dot(L,L);
}

float3 evaluateLight( float3 pos, float3 normal )
{
    float3 lightPos = LPOS();
    float3 L = lightPos-pos;
    float distanceToL = length(L);
    float3 Lnorm = L/distanceToL;
    return max(0.0,dot(normal,Lnorm)) * evaluateLight(pos);
}

// todo: need some 3d noise to make the 'cloud' density
//// To simplify: wavelength independent scattering and extinction
void getParticipatingMedia(thread float& sigmaS, thread float& sigmaE, float3 pos, float time)
{
//    float cloud = 5.0 * clamp( fbm(pos * 0.05), 0.0, 1.0 );
//    cloud *= smoothstep(1.0, 0.5, -time+pos.y);
//    cloud *= smoothstep(-1.0, -0.5, -time+pos.y);
//    cloud *= smoothstep(1.0, 0.0, pos.z);
//    cloud *= smoothstep(-1.0, -0.5, pos.z);
//    cloud = 0.0;
//    pos += float3(0,-30,0);
    float heightFog = 3.0 + 3.0*clamp(fbm( pos*0.5 ),0.0,1.0);
    heightFog = 0.095*clamp((heightFog-pos.y)*1.0, 0.0, 1.0);
    
//    float NoiseScale = 3.74;
//    float NoiseIsoline = 1.02;
//    float3 p2 = pos / NoiseScale + time * float3(0.5);
//        p2 *= 1.24;
    
    // 1 => sizeSignal(time)
    
//    float noise = NoiseScale * (fbm(p2 + 0.422*fbm( p2 )) - NoiseIsoline);
//    noise *= smoothstep(1.0, 0.5, noise);
//    noise *= smoothstep(-1.0, -0.5, noise);
//    float cloudFog = 0.095 * clamp (noise, 0.0, 1.0);
//    const float fogFactor = 5.5;
    
    const float constantFog = 0.001;

    sigmaS = constantFog + heightFog;

    const float sigmaA = 0.0;
    sigmaE = max(0.000000001, sigmaA + sigmaS); // to avoid division by zero extinction
}

// henyey-greenstein
float phg(float ang, float g) {
    float pi = 3.141592653;
    float g2 = pow(g, 2.0);
    return (1. - g2) / (4.*pi*pow(1.0 + g2 - 2.*g*cos(ang), 1.5));
}

float phaseFunction( float ang )
{
    return mix(phg(ang, 0.01), phg(ang, 0.99), 0.2);
}


float volumetricShadow(float3 from, float3 to, float time)
{
    const float numStep = 16.0; // quality control. Bump to avoid shadow alisaing
    float shadow = 1.0;
    float sigmaS = 0.0;
    float sigmaE = 0.0;
    float dd = length(to-from) / numStep;
    for(float s=0.5; s<(numStep-0.1); s+=1.0)// start at 0.5 to sample at center of integral part
    {
        float3 pos = from + (to-from)*(s/(numStep));
        getParticipatingMedia(sigmaS, sigmaE, pos, time);
        shadow *= exp(-sigmaE * dd);
    }
    return shadow;
}

float2 raycast (float3 ro, float3 rd, thread float4& scatTrans, float time){
    float2 res = float2(-1.0,-1.0);

    float tmin = 0.0021;
    tmin = 1.0;
    float tmax = 150.0;
    
    float eps = 0.001;
    float t = tmin;
    
    float sigmaS = 0.0;
    float sigmaE = 0.0;

    float transmittance = 1.0;
    float3 scatteredLight = float3(0.0, 0.0, 0.0);
    float3 lightPos = LPOS();
    
    //2052
    for( int i = 0; i < 2512 && t < tmax; i++) {
        float3 p = ro + rd*t;
        float2 h = map( p, time );
        float dd = h.x;
        
        getParticipatingMedia(sigmaS, sigmaE, p, time);
        
        for (int i = 0; i < 3; i++ )
        {
            float sigmaS2 = sigmaS * pow(1.83, float(i));
            float sigmaE2 = sigmaE * pow(2.22, float(i));
            float ang = acos(dot(normalize(lightPos), rd)) * pow(1.514, float(i));
            float3 S = evaluateLight(p) * sigmaS2 * phaseFunction(ang) * volumetricShadow(p,lightPos,time);// incoming light
            float3 Sint = (S - S * exp(-sigmaE2 * dd)) / sigmaE2; // integrate along the current step segment
            scatteredLight += transmittance * Sint; // accumulate and also take into account the transmittance from previous steps

        }

//         Evaluate transmittance to view independentely
        transmittance *= exp(-sigmaE * dd);
        
        if( abs(h.x) < eps){
            res = float2(t, h.y);
            break;
        }

        t += h.x; //12
    }

    // return scattering, transmittance (inout equivalent scatTran)
    scatTrans = float4(scatteredLight, transmittance);
    return res;
}

/*
 
                                                                                             
                                                                                             
 `7MMF'   `7MF'MMP""MM""YMM `7MMF'`7MMF'      `7MMF'MMP""MM""YMM `7MMF'`7MM"""YMM   .M"""bgd
   MM       M  P'   MM   `7   MM    MM          MM  P'   MM   `7   MM    MM    `7  ,MI    "Y
   MM       M       MM        MM    MM          MM       MM        MM    MM   d    `MMb.
   MM       M       MM        MM    MM          MM       MM        MM    MMmmMM      `YMMNq.
   MM       M       MM        MM    MM      ,   MM       MM        MM    MM   Y  , .     `MM
   YM.     ,M       MM        MM    MM     ,M   MM       MM        MM    MM     ,M Mb     dM
    `bmmmmd"'     .JMML.    .JMML..JMMmmmmMMM .JMML.   .JMML.    .JMML..JMMmmmmMMM P"Ybmmd"
                                                                                             
                                                                                             
 
*/
float3 palette( float t, float3 a, float3 b, float3 c, float3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

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
        if( h<0.00001 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h/2.0;
    }
    return res;
}

/*
 
                                                                                                            
                                                                                                            
 `7MM"""Mq.  `7MM"""YMM  `7MN.   `7MF'`7MM"""Yb. `7MM"""YMM  `7MM"""Mq.  `7MMF'`7MN.   `7MF' .g8"""bgd
   MM   `MM.   MM    `7    MMN.    M    MM    `Yb. MM    `7    MM   `MM.   MM    MMN.    M .dP'     `M
   MM   ,M9    MM   d      M YMb   M    MM     `Mb MM   d      MM   ,M9    MM    M YMb   M dM'       `
   MMmmdM9     MMmmMM      M  `MN. M    MM      MM MMmmMM      MMmmdM9     MM    M  `MN. M MM
   MM  YM.     MM   Y  ,   M   `MM.M    MM     ,MP MM   Y  ,   MM  YM.     MM    M   `MM.M MM.    `7MMF'
   MM   `Mb.   MM     ,M   M     YMM    MM    ,dP' MM     ,M   MM   `Mb.   MM    M     YMM `Mb.     MM
 .JMML. .JMM..JMMmmmmMMM .JML.    YM  .JMMmmmdP' .JMMmmmmMMM .JMML. .JMM..JMML..JML.    YM   `"bmmmdPY
                                                                                                            
                                                                                                            
    lighting, raymarching.
*/

float2 csqr( float2 a )  { return float2( a.x*a.x - a.y*a.y, 2.*a.x*a.y  ); }

float3x3 setCamera( float3 ro, float3 ta, float cr )
{
    float3 cw = normalize(ta-ro);
    float3 cp = float3(sin(cr), cos(cr),0.0);
    float3 cu = normalize( cross(cw,cp) );
    float3 cv =          ( cross(cu,cw) );
    return float3x3( cu, cv, cw );
}

float3 pattern(float3 rd, float time) {
    rd*=2.5;
    rd = rd + float3(0.15*time,0,0);
    return max(1.91*floor(64.0*(cos(0.75*snoise(0.5+0.5*sin(float3(0.5,1.0,0.0)*time*0.2+rd*5)+rd*0.5))))/64.0 - 1.0, 0.0)*0.2*float3(0.923, 0.802, 0.48);
}

// triplanar mapping
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

float3 render(float3 ro, float3 rd, float3 rdx, float3 rdy, float time, array<texture2d<float, access::sample>, 20> ground, texturecube<float> cubemap, sampler defaultSampler)
{
//    float3 transparent = cubemap.sample(defaultSampler, rd).bgr;
    float3 col = float3(1.0);
//    float3 col = float3(0);
    float4 scatTrans = float4(0.0);
    float2 res = raycast(ro,rd,scatTrans,time); // traceScene equivalent

    float t = res.x;
    float m = res.y;

    float3 pos = ro + rd*t;
    float3 nor = calcNormal(pos, time);
    
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
            scale = 0.5;
        } else if ( m > 10. ) {
            scale = 0.004;
        }
        nor = doBumpMap(defaultSampler, ground[texIndex+1.0], texScale, pos, nor, scale);
    }
    
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
    
    // lighting
    if ( m > 0. && m < 20. ) {
        float3 l = float3(0.0);
//            material material;
        
        float bou = clamp( dot(nor, float3(0.0, -0.5, 0.5)), 0.0, 1.0 );
        
        float occ = calcAO(pos, nor, time);
        
//            float3 rfrColor = cubemap.sample(defaultSampler, rfr).bgr;

        // indoor lighting
        
        // top - BRDF
        {
            float3 lig = LPOS();

            float3 rfl = reflect(rd, nor);
            
            float dif = clamp(dot(lig,nor), 0.0, 1.0);
            float difrd = clamp(dot(-rd, nor), 0.0, 1.0);
//                dif *= occ;

            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);

            float shadow = 1.0;//calcSoftshadow(pos, lig, 0.0021, 20.0, 4.0, time);
            float3 clr = 8.0*normalize(float3(0.5, 0.633, 1.1));
            
            // fresnel
            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            
            // distribution
            float a2 = pow(a, 2.0);
            float dTr = a2 / ( 3.14159*pow((pow(dot(hNor, nor), 2.0) * (a2 - 1.0) + 1.0), 2.0) );

            float k = pow((rou + 1.0)/2.0, 2.0) / 8.0;
            float G1L = dif / ( dif * (1.0-k) + k );
            float G1V = difrd / ( difrd * (1.0-k) + k );
            
            float GSmith = G1L * G1V;
            
//            env = pow(env, 0.5);
//            env = 1.0;
            
            // full spectral light addition
            float3 spe = clamp((fSch * dTr * GSmith) / ( 4.0 * dif * difrd ), 0.0, 1.0);
            
            l+= spe * clr * occ * shadow;  // spec - add material, or not? shadow, or not?

            l+= (0.5 + (1.0-rou)*dif) * shadow * clr * occ * material; // dif
            
//            l += 0.4 * bou * rou * material;

        }
        
        {
            float3 lig = normalize(float3(0.8, 0.1, 0.56));
            float dif = clamp(dot(lig,nor), 0.0, 1.0);
            float difrd = clamp(dot(-rd, nor), 0.0, 1.0);
//                dif *= occ;

            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);

            float shadow = 1.0;//calcSoftshadow(pos, lig, 0.021, 10.0, 4.0, time);
            
            float3 clr = 7.0*normalize(float3(1.0, 0.6, 0.5));
            
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
            l+= spe * clr * occ * shadow; // spec - add material, or not? shadow, or not?

            l+= ((1.0-rou)*dif + 0.5) * clr * occ * shadow * material; // dif
            
//            float bounce = abs(dot(-1.0 * lig, hal));
//            l+= (0.15 * material * rou * bounce);
        }
    
        col = ((0.05 * material) + (0.95 * l));
    }
    
    col = scatTrans.rgb + scatTrans.a * col;
    return float3( clamp(col, 0.0, 1.0) );
    
}

kernel void compute(texture2d<float, access::write> output [[texture(0)]],
                    texturecube<float> cubemap [[texture(1)]],
                    array<texture2d<float, access::sample>, 20> ground [[texture(2)]],
                    sampler defaultSampler [[sampler(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    constant Uniforms &uniforms [[buffer(0)]])
{
    float change = -uniforms.time;
    change = 0.0;
    float3 ro = float3( 0.0, change+4.2, 10.5);
    float3 ta = float3( 0.0, change+4.09, 0.5);
    float3x3 ca = setCamera(ro, ta, 0.0);
    float3 total = float3(0.0);
    
    int width = output.get_width();
    int height = output.get_height();
    float aspect = uniforms.resolution[0] / uniforms.resolution[1];
    float2 uv = float2(gid) / float2(width, height);
    
    uv = uv * 2.0 - 1.0;
    uv.x *= aspect;
    uv.y *= -1.0;
    
    float3 rd = ca * normalize( float3(uv, 2.2) );

    // ray differentials
    float2 px =  (uv+float2(1.0,0.0) - float2(0.5));
    float2 py =  (uv+float2(0.0,1.0) - float2(0.5));
    float3 rdx = ca * normalize( float3(px, 2.5));
    float3 rdy = ca * normalize( float3(py, 2.5));
    
    float3 color = render( ro, rd, rdx, rdy, 0.5*uniforms.time, ground, cubemap, defaultSampler );

    color = pow(color, float3(0.4545));

    total += color;
    total = min(total, 1.0);

    // dithering
    total += (1.0/100.0) * sin(uv.x*944.111)*sin(uv.y*843.32);
    
    output.write(float4(total, 1.0), gid);
}
