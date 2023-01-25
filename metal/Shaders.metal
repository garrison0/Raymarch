#include <metal_stdlib>
#include "Shaders.h"
using namespace metal;

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
float2 mapFlux(float3 p, float distToBound, float time) {
    float2 res = float2(1e10, 0.0);

    float3 p3 = p;
    float bounds = (smoothstep(0.15, -0.5, distToBound));// * smoothstep(-0.68, -0.675, p.y) * smoothstep(1.75, -0.15, p.y);
    
    {
//        float NoiseScale = 2.52391;
//        float NoiseScale = 8.0;
        float NoiseScale = 0.35;
//        float NoiseIsoline = 0.23319 * bounds;
        float NoiseIsoline = 0.5 * (bounds);
        time *= 0.05;

        p3 = p3 + time * float3(-2.0,0.0,0);
        p3 = p3 / NoiseScale;
        
        // rotate
//        float3 q = p;
        float k = 5.0*time;
        float c = cos(k*p.y);
        float s = sin(k*p.y);
        float2x2 m = float2x2(c, -s, s, c);
        float3 q = float3(m*(p.xz), p.y);
        
        // twist
//        k = 0.1;
//        c = cos(k*q.y);
//        s = sin(k*q.y);
//        m = float2x2(c, -s, s, c);
//        q = float3(m*q.xz, q.y);
//        q = q / NoiseScale;
        
        // the real one:
//        float noise = NoiseScale * (1.9*fbm(2.44*fbm(float3(1.5,3.0,1.5)*q*0.1 + 0.99*q.x) + 0.5*(0.5+0.5*sin(sin(2.965*q))))
//        - NoiseIsoline);
        
        // for testing water render:
//        float noise = NoiseScale * (fbm( float3(1.0,0.5,0.5)*q + fbm ( 4.0*time+1.0*sin(q) + fbm(0.5*time+0.5*q))) - NoiseIsoline);
//        float noise = sdExtrudedTriangle(p, float2(3.0, 2.5), 0.5);
        float noise = sdExtrudedTriangle(q, float2(2.0, 2.0), 0.5);
        res = opU(res, float2(noise, 15.0));
    }

    return res;
}

float2 map(float3 p, float time)  {
    float2 res = float2(1e10, 0.0);

//    return float2(sdfSphere(p + float3(0,0,1), 0.5), 15.0);
//    float dToBound = sdCappedCylinder(p + float3(0,-4.25,0.0), float3(-10,0,0), float3(3,0,0), 0.65);
    float dToBound = sdExtrudedTriangle(p + float3(-2.0, -4.5, 0), float2(0.8, 10.0), 0.5) - 0.2;
    
//    time *= 0.64;
//    time = time + (0.5+0.5*cos(sin(time)));
//    p.y -= 4.25;
//    float k = 0.5 + 3.0*(0.5+0.5*cos(cos(time)+0.5*fbm(cos(p+time*0.8)+0.8*time+p*0.9)));
//    float c = cos(k*p.y + time + 0.44+0.44*sin(cos(0.2*time + p.y*(0.4+0.4*cos(p.y)))));
//    float s = sin(k*p.y + time + 0.44+0.44*cos(cos(0.1*time + p.y)));
//    float2x2 m = float2x2(c, -s, s, c);
//    float3 q = float3(m*(p.xz - float2(0.,5.)), p.y);
//
//    float dToBound = sdBox(p, float3(1.05, 1.35, 1.05));
//    res = opU(res, float2(dToBound, 10.0));
//    if (dToBound > 0.001) {
//        res = opU(res, float2(dToBound, 10.0));
//    } else {
        float2 dFlux = mapFlux(p + float3(0.0,-4.5,0.0), dToBound, time);
//        dFlux.x -= 0.01;
        res = opU(res, dFlux);
//    }
    
    return res;
}

float2 raycast (float3 ro, float3 rd, float time){
    float2 res = float2(-1.0,-1.0);

    float tmin = 0.0021;
    float tmax = 100.0;
    
    float eps = 0.001;
    float t = tmin;
    //2052
    for( int i = 0; i < 2052 && t < tmax; i++) {
        float2 h = map( ro + rd*t, time );

        if( abs(h.x) < eps){
            res = float2(t, h.y);
            break;
        }

        t += h.x/12.0; //12
    }

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

float3 pattern(float rdY, float time) {
    return (0.5+0.5*cos(rdY*5.22))*0.8*float3(0.996, 0.972, 0.964);
}

float3 render(float3 ro, float3 rd, float3 rdx, float3 rdy, float time, texturecube<float> cubemap, sampler defaultSampler)
{
    float3 transparent = cubemap.sample(defaultSampler, rd).bgr;
    transparent = pow(transparent, 1.0/0.4545);
    float3 col = 0.1*transparent + 0.9*pattern(rd.y, time);
    float2 res = raycast(ro,rd, time);

    float t = res.x;
    float m = res.y;

    float3 pos = ro + rd*t;
    float3 nor = calcNormal(pos, time);
    float3 material = float3(0);
    float3 f0 = float3(0.05);
    float a = 1.0;

    // MATERIALS
    // 15 - water
    if (m < 16.) {
        material = pow(float3(0.47, 0.83, 0.97), 2.0);
        f0 = material;
    }
    
    // lighting
    if ( m > 0. ) {
        float3 l = float3(0.0);
        if (m < 20) {
            material *= 0.72 * material;
            
            float bou = clamp( 0.3-0.7*nor.y, 0.0, 1.0 );
            float3 rfl = reflect( rd, nor );
            
            float refractionIndex = 1.33;
            float3 rfrrd = -refract( rd, nor, refractionIndex); // 1.31 = ice, 1.40 = kerosene
            
            float3 rfrro = pos + 5.0*rfrrd;
            float rfrdist = raycast(rfrro, -rfrrd, time).x;
            
            
            float3 rfrpos = rfrro + rfrdist * rfrrd;
            float3 rfrnormal = calcNormal(rfrpos, time);
            
            float3 rfrColor = cubemap.sample(defaultSampler, refract(-rfrrd, rfrnormal, 1.0/refractionIndex)).bgr;
            rfrColor = pow(rfrColor, 1.0/0.4545);

            float3 rflColor = cubemap.sample(defaultSampler, rfl).bgr;
            rflColor = pow(rflColor, 1.0/0.4545);
//            float3 rfrColor = cubemap.sample(defaultSampler, rfr).bgr;
            
            
//            l = rflColor;
            
            // percent refracted = 1 - fresnel
            
            // get water color!
//            1. bump map the normal, if wanted
//            2. get refract and reflect directions
//            float3 normalDelta = NORMALMAP_main(ptIce*ROUGHNESS,iceSurface.normal)*ROUGHNESS/10.;
//            vec3 iceSurfaceNormal = normalize(iceSurface.normal+normalDelta);
//            vec3 refract_dir = refract(dir,iceSurfaceNormal,1.0/REFRACTION_IDX); //Ice refraction index = 1.31
//            vec3 reflect_dir = reflect(dir,iceSurfaceNormal);
//
//            //Trace reflection
//            iceData.reflectTraceData = TRACE_reflexion(ptIce,reflect_dir);
//
//            //Balance between refraction and reflection (not entirely physically accurate, Fresnel could be used here).
//            float fReflectAlpha = 0.5*(1.0-abs(dot(normalize(dir),iceSurfaceNormal)));
//            iceData.reflectTraceData.alpha = fReflectAlpha;
//            vec3 ptReflect = ptIce+iceData.reflectTraceData.rayLen*reflect_dir;
//
//            //Trace refraction
//            iceData.translucentTraceData = TRACE_translucentDensity(ptIce,refract_dir);
//
//            vec3 ptRefract = ptIce+iceData.translucentTraceData.rayLen*refract_dir;
//            vec3 exitRefract_dir = refract(refract_dir,-iceData.translucentTraceData.normal,REFRACTION_IDX);
//
//            //This value fades around total internal refraction angle threshold.
//            if(length(exitRefract_dir)<=0.95)
//            {
//                //Total internal reflection (either refraction or reflexion, to keep things cheap).
//                exitRefract_dir = reflect(refract_dir,-iceData.translucentTraceData.normal);
//            }
//
//            //Trace environment upon exit.
//            iceData.exitTraceData = TRACE_cheap(ptRefract,exitRefract_dir);
//            iceData.exitTraceData.matID = MATERIALID_FLOOR;
//
//            return iceData;
            // indoor lighting
            // top - BRDF
            {
                float3 lig = normalize(float3(0.5,0.88,0.4));

                float3 hal  = normalize(lig - rd);
                float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
                
                float3 clr = normalize(float3(0.5, 0.633, 0.9));

                // fresnel
//                float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
                
                fre = pow(fre, 5.0);
                float3 refrfl = 0.64*rflColor*fre + 0.64*rfrColor*(1.0-fre);
                l += refrfl + 0.1 * transparent;
            }
            //side
            {
                float3 lig = normalize(float3(-0.6, 0.03, 0.1));
                

                float3 hal  = normalize(lig - rd);
                float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
                
                float3 clr = normalize(float3(1.0, 0.6, 0.5));

                // fresnel
//                float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);

                fre = pow(fre, 5.0);
                float3 rflrfr = 0.64*rflColor*fre + 0.64*rfrColor*(1.0-fre);
                l += rflrfr + 0.1 * transparent;
                
                float bounce = abs(dot(-1.0 * lig, hal));
                
//                l += rflColor * 2.0 * fre * fre;
                l+= (0.04 * material * bounce);
            }
        }
        
        col = ((0.02 * material) + (1.0 * l));
    }
    
    return float3( clamp(col, 0.0, 1.0) );
    
}

kernel void compute(texture2d<float, access::write> output [[texture(0)]],
                    texturecube<float> cubemap [[texture(1)]],
                    sampler defaultSampler [[sampler(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    constant Uniforms &uniforms [[buffer(0)]])
{
    float3 ro = float3( 0.0, 4.5, 7.5);
    float3 ta = float3( 0.0, 4.41, 0.5);
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
    
    float3 color = render( ro, rd, rdx, rdy, 1.15*uniforms.time, cubemap, defaultSampler );

    color = pow(color, float3(0.4545));

    total += color;
    total = min(total, 1.0);

    // dithering
    total += (1.0/100.0) * sin(uv.x*944.111)*sin(uv.y*843.32);
    
    output.write(float4(total, 1.0), gid);
}
