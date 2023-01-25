#include <metal_stdlib>
#include "ShaderTypes.h"
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

//__constant float2x2 m2 = float2x2( 0.60, -0.80, 0.80, 0.60 );

float fbm( float3 p ) {
    float3x3 m3 = float3x3( 0.00,  0.80,  0.60,
                           -0.80,  0.36, -0.48,
                           -0.60, -0.48,  0.64 );
    float f = 0.0;
    f += 0.5000*noise( p ); p = m3*p*2.02;
    f += 0.2500*noise( p ); p = m3*p*2.03;
    f += 0.1250*noise( p ); p = m3*p*2.01;
    f += 0.0625*noise( p );
    return f/0.9375;
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

float turbulence( float3 p ) {
    float t = -.5;

    for (float f = 1.0 ; f <= 10.0 ; f++ ){
        float power = pow( 2.0, f );
        t += abs( snoise( float3( power * p ) ) / power );
    }

    return t;
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

// vertical
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
    float4 q = float4(cos(theta / 2.0), sin (theta / 2.0) * n);
    float3 temp = cross(q.xyz, p) + q.w * p;
    float3 rotated = p + 2.0*cross(q.xyz, temp);
    return rotated;
}

float2 mapFlux(float3 p, float time, float3 pos) {
    float2 res = float2(1e10, 0.0);
    // time *= 0.;
    float3 extents = 2.0*float3(0.6, 0.3, 0.7);
    float3 l = float3(5, 2, 2);

    // piece 2: really nice 'plastic bag' effect. this should be a piece (maybe -the- piece)
    // make sure to remove the bounds (in 'map')
    {
//        time *= 1.0;
//        float NoiseScale = 3.0;
//        float NoiseIsoline = 0.3;
//        float3 p2 = p / NoiseScale + time * float3(0.5);
//        float noise = (1.0+0.25*sizeSignal(time)) * NoiseScale * (fbm( p2 ) - NoiseIsoline);
//        p.x -= noise;
        
        time *= 1.1;
        float NoiseScale = 2.8;
        float NoiseIsoline = 0.29;
        float3 p2 = p / NoiseScale + time * float3(0.5);
        float noise = (1.25+0.25*sizeSignal(time)) * NoiseScale * (fbm( p2*1.2 ) - NoiseIsoline);
        p.x -= noise;

        // make same height for IWO
        float d = sdCapsule(p, float3(-2.5, 0.0, 0.0), float3(2.5,2.0,0.0)).x - ( 0.35 + 0.65 * ( sizeSignal(time)) ); // capsule // two capsules alternating?
        res = opSmoothU(res, float2(d, 15.0), 2.25);
    }

    return res;
}

float2 map (float3 p, float time) {
    float2 res = float2(1e10, 0.0);
    float boxDistance = 5.0;

    // walls, ground
    // res = float2(sdBox(rotatePoint(p + float3(boxDistance, 0.0, 20.0), float3(0,1,0), 3.141592653/4.0), float3(0.5, 10.0, 20.0)), 25.0);
    // res = opU(res, float2(sdBox(rotatePoint(p + float3(-boxDistance, 0.0, 20.0), float3(0,1,0), -3.141592653/4.0), float3(0.5, 10.0, 20.0)), 26.0));
    res = opU(res, float2(sdBox(p + float3(0, 0.0, 0.0), float3(100.0, .1, 70.0)), 30.0));
    
    // kettle
    // res = opU(res, mapPot(p + float3(0.0, -1.15, 15.0), time));
    
    // float dToBound = sdBox(p + float3(0,-4.15,15.0), float3(3, 3, 3));
    // float dToBound = sdfSphere(p+ float3(0.0, -2.15, 15.0), 1.0);
    // float eps = 0.001;
    // p.z = max(-p.z, 40.0);

        // float3 pos = float3(0.0, -3.5, 10.0);
        // float dFlux = opIntersection(mapFlux(p, time, p+pos).x, sdfSphere(p + pos, 5.0));
        // res = opU(res, float2(dFlux, 30.0));

    float3 pos = float3(0.0, -0.5, 10.0);
    // iwo: x -> -3.2
    res = opU(res, mapFlux(p - float3(-1.0,3.0,-10), time, p+pos));

    return res;
}

float2 raycast (float3 ro, float3 rd, float time){
    float2 res = float2(-1.0,-1.0);

    float tmin = 0.00001;
    float tmax = 120.0;
    
    // raytrace floor plane
    // float tp1 = (-ro.y)/rd.y;
    // if( tp1 > 0.0 )
    // {
    //     tmax = min( tmax, tp1 );
    //     res = float2( tp1, 0.0 );
    // }

    float eps = 0.001;
    float t = tmin;
    for( int i = 0; i < 528 && t < tmax; i++) {
        float2 h = map( ro + rd*t, time );

        if( abs(h.x) < eps){
            res = float2(t, h.y);
            break;
        }

        t += h.x/3.0; // * 0.95;
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
    const float eps = 0.0001;
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


float3 skyColor( float3 ro, float3 rd, float3 sunLig, float time )
{
    float3 col = float3(0.8);
    // float t = (20.0-ro.y)/rd.y;
    // if( t>0.0 )
    // {
    //     float3 pos = ro + t * rd;
    //     pos = pos * 0.003;
    //     col = float3(vnoiseOctaves(float2(0.0, -uTime/50.) + pos.xz, 1.0, 0.5));
    //     col = 0.1 + 0.6 * col + 0.3 * float3(fbm(0.2*pos + float3(-uTime/100., -uTime / 75., uTime/25.)));
    //     col *= (1.0 / (pos.z*pos.z));
    // }
    
    return col;
}

float2 csqr( float2 a )  { return float2( a.x*a.x - a.y*a.y, 2.*a.x*a.y  ); }


float mapMarble( float3 p, float time ) {
    float res = 0.;
    
    // add back for iwo
//    float NoiseScale = 3.0;
//    float NoiseIsoline = 0.3;
//    float3 p2 = p / NoiseScale + time * float3(0.5);
//    float noise = (1.0+0.25*sizeSignal(time)) * NoiseScale * (fbm( p2 ) - NoiseIsoline);
//    p.x -= noise;

    p *= 0.17;
    float3 c = p;
    // c.y *= 2.0;
    for (int i = 0; i < 20; ++i) {
        p =.7*abs(p)/dot(p,p) -.7;
        p.yz= csqr(p.yz);
        p=p.zxy;
        res += exp(-40. * abs(dot(p,c)));
        
    }
    return res/2.;
}

float3 marchMarbleColor(float3 ro, float3 rd, float tmin, float tmax, float time, int option) {
    float t = tmin;
    float dt = .02;
    float3 col= float3(0.);
    float c = 0.;
    for( int i=0; i<48; i++ )
    {
        t+=dt*exp(-2.*c);
        if(t>tmax)break;
        
        c = mapMarble(ro+t*rd, time);
        
        //col = .99*col+ .08*float3(c*c, c, c*c*c);//green
        if (option < 1) {
            col = .93*col+ .08*float3(0.0, c*c, c);//blue
        } else {
            col = .95*col+ .08*float3(c*c, c*c*c, c);//blue
        }
        
    }
    return col;
}

float3 skyColorRef( float3 ro, float3 rd, float3 sunLig, float time )
{
    float3 col = float3(0.3,0.4,0.56)*0.3 - 0.3*rd.y;

    float t = (20.0-ro.y)/rd.y;
    if( t>0.0 )
    {
        float3 pos = ro + t * rd;
        pos = pos * 0.02;
//        pos.x *= 2.0;
        col = float3(vnoiseOctaves(float2(0.0, -time/50.) + pos.xz, 1.0, 0.5));
        col = 0.1 + 0.5 * col + 0.3 * float3(fbm(0.2*pos + float3(-time/100., -time / 75., time/25.)));
        col.g *= col.g;
        col.r *= 1.5;
        col.b *= 0.5;
        col *= (0.1 / (pos.y));
    }
    
    float sd = pow( clamp( 0.04 + 0.96*dot(sunLig,rd), 0.0, 1.0 ), 4.0 );
    
    // over time:
    // set to -abs((60-55*sd))
    // col = mix( col, float3(0.2,0.25,0.30)*0.7, exp(-40.0*rd.y) ) ;

    col = mix( col, float3(1.0,0.30,0.05), sd*exp(-abs((16.0-(12.05*0.5*sd)))*rd.y) ) ;
    col = mix( col, float3(0.2,0.25,0.34)*0.7, exp((-24.0)*(0.0875+rd.y)) ) ;

    return col;
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
    float3 col = float3(0.0);

//    col = float3(0.92);
    col = skyColorRef(ro, rd, float3(8.0, -1.1, -8.0), time);

    float2 res = raycast(ro, rd, time);
    float t = res.x;
    float m = res.y;

    float3 pos = ro + rd*t;
    float3 normal = calcNormal(pos, time);

    float3 material = float3(0);
    float2 K = float2(0,1); // amount, power for specular
    float3 f0 = float3(0.05);
    float rou = 0.0; // roughness
    float3 hNor = float3(0); // microfacet normal - is getting it from texture better than 'hal'?
    float a = 1.0;
    
    float3 texWeights = abs(normal) - 0.2;
    texWeights = pow(texWeights, 2.0);
    texWeights = saturate(texWeights);
    texWeights.xy = pow(texWeights.xy, 1.5);
    texWeights = normalize(texWeights);
    int texIndex = 0;
    float scale = 1.0;

    // MATERIALS
    // 25 / 26 - walls
    // 30 - ground

    // 14 - noise (silicon-ish)
    // kettle - metal
    if (m < 16.) {
        // if (m < 15.) {
        //     // spout - add detail
        //     angleBetweenXY = asin(nor.x) / 3.141592653;
        //     angleBetweenXZ = asin(nor.y) / 3.141592653;
        //     texCoords = 0.8 * float2(angleBetweenXY, angleBetweenXZ);
        //     material = 0.6 *texture(uMetalColor, 0.7 * pos.xy + 0.7*texCoords).rgb
        //             +  0.4 *texture(uMetalColor, 0.62 * pos.yz + 0.6*texCoords).rgb;
        //     hNor = texture(uMetalNormal, 0.82 * pos.xy + 0.65*texCoords).rgb;
        //     rou = texture(uMetalRough, 0.82 * pos.xy + 0.65*texCoords).r;
        // } else {
        
        float tmax = fbm(float3(1.2,1.2,1)*pos);
        material = marchMarbleColor(ro + float3(-1,-1,0.0) + smoothstep(0,30,time)*0.1*time*float3(0,-0.15,-0.15), rd, t, t + 4.99*tmax, time, 0);
        // material = 0.65*texture(uMetalColor, 0.35 * pos.xz + texCoords).rgb
        //     +  0.35*texture(uMetalColor, 0.212 * pos.xy + texCoords).rgb;
//        hNor = texture(uMetalNormal, 0.7*texCoords).rgb;
        
        
        float3x3 mats = float3x3(ground[texIndex+2].sample(defaultSampler, pos.yz).rgb,
                     ground[texIndex+2].sample(defaultSampler, pos.xz).rgb,
                     ground[texIndex+2].sample(defaultSampler, pos.xy).rgb);
        hNor += scale * mats * texWeights;
    
        float3 mats1x3 = float3(ground[texIndex+2].sample(defaultSampler, pos.yz).r,
                      ground[texIndex+2].sample(defaultSampler, pos.xz).r,
                      ground[texIndex+2].sample(defaultSampler, pos.xy).r);
    
        rou += scale * dot(texWeights, mats1x3);
        
//        rou = dot(nor, float3(texture(uMetalRough, pos.zy).r,
//                    texture(uMetalRough, pos.xz).r,
//                    texture(uMetalRough, pos.xy).r));
        // rou = texture(uMetalRough, 0.7*texCoords).r;
        // rou = 1.0;
        // }

        K = float2(0.35, 16.0);
//        f0 = float3(.972, .961, .915);
        f0 = float3(0.2);
        a = 16.0 * (0.15 + 0.85 * (1.0 - rou));
    // wall - left
    } else if (m < 26.) {
        K = float2(0.05, 2.0);
        material = float3(.9608, .9686, .949);
    // wall - right
    } else if (m < 27.) {
        K = float2(0.05, 2.0);
        material = float3(.9608, .9686, .949);
    // floor
    } else if (m < 31.) {
        K = float2(0.25, 4.0);
        material = float3(0.8, 0.749, 0.7019);
    }

    // lighting
    if ( m > 0. ) {
        material *= 0.72 * material;
        
        float occ = calcAO(pos, normal, time);
        a = 0.5*a + 0.5*((2.0 / pow(rou, 2.0)) - 2.0);
        
        float bou = clamp( 0.3-0.7*normal.y, 0.0, 1.0 );
        float3 ref = reflect( rd, normal );
        float3 lin = float3(0);
        float3 secLin = float3(0);
        // indoor lighting
        // top - BRDF
        {
            float3 lig = normalize(float3(0.2,1.0,0.05));

            float dif = clamp(dot(lig,normal), 0.0, 1.0);
            dif *= occ;

            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
            // fre = 0.05 * fre + 0.95 * clamp(1.0 - dot(lig, hNor), 0.0, 1.0); // i like both qualities
            // set min back to 0.000001;
//            float shadow = calcSoftshadow(pos, lig, 0.000001, 100.0, 16.0, time);
            float shadow = 1.0;
//            (m == 16.5) ? shadow = 1.0 : shadow = shadow;

            float3 clr = normalize(float3(0.5, 0.633, 0.9));
            // float speBias = smoothstep(0.3, 0.42, ref.y); // to make it seem more like an area light
            // float speBias = 1.0; // or not
            
            // fresnel
            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            
            // distribution
            // float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(nor, hNor), 0.0, 1.0), a ); // more fake, 90s
            float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(normal, hNor), 0.0, 1.0), a); // more accurate - K.y is normally a

            // full spectral light addition
            float3 spe = (fSch * dBlinnPhong) / 2.0;
            lin += K.x * 0.65 * material * spe * dif * clr * shadow; // spec - add material, or not? shadow, or not?

            if (m == 30) {
                float fre2 = 0.35+0.7*dot(rd, float3(0,0,-1));
                secLin += 1.0*pow(fre2,5.0)*skyColorRef(pos, ref, float3(8,-1,-8), time);
//                secLin += 0.5*ground[0].sample(defaultSampler, pos.xz).rgb;
            }
            
            lin += (1.0 - K.x) * dif * shadow * material; // dif
            lin += K.x * material * spe * dif * shadow; // spec - add material, or not? shadow, or not?
//            float fre2 = pow(.5+ clamp(dot(normal,rd),0.0,1.0), 3. )*1.3;
//            lin += fre2*skyColor(pos, ref, float3(8,-1,-8), time);
        }
        //side
        {
            float3 lig = normalize(float3(-0.53, 0.35, 0.05));
            float dif = 0.01 + 0.99 * clamp(dot(lig, normal), 0.0, 1.0);
//            float shadow = calcSoftshadow(pos, lig, 0.0021, 20.0, 9, time);
            float shadow = 1.0;
//            (m == 16.5) ? shadow = 1.0 : shadow = shadow;
            // shadow = pow(shadow, 0.5);
            float3 clr = float3(1.0, 0.6, 0.5);
            dif *= occ;

            float3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, normal), 0.0, 1.0);

            float3 spe = float3(1)*(pow(clamp(dot(normal,hal), 0.0, 1.0), a / 2.0));
            // spe *=

            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            spe *= fSch;

            // float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(nor, hal), 0.0, 1.0), a);

//             float3 spe = (fSch * dBlinnPhong) / (4.0);
            
            if (m == 30) {
//                secLin += 0.75*pow(fre, 5.0)*skyColorRef(pos*4.0, ref, float3(8,-1,-8), time + 44.21);
                float fre2 = 0.35+0.7*(1.0-abs(rd.y));
                secLin += 0.75*pow(fre2,5.0)*skyColorRef(pos*4.0, ref, float3(8,-1,-8), time + 44.21);
                float3 pos3 = pos + time*0.073*float3(0.1,0.0,-0.4);
//                pos += time*float3(0.1,0.0,-0.4);
                secLin += 0.005*marchMarbleColor(0.2*pos3, ref, 0.5, 10.0, time, 1);
            }
            else {
                lin += K.x * 0.75 * dif * spe * material * shadow;
                lin += (1.0 - K.x) * 3.0 * dif * shadow * material;
                
            }
            
            
            
        }
        // back (?)
        // below - bounce
        {
//            float spoutFix = clamp( dot(normal, normalize(float3(-0.3, -0.5, 0.1))), 0.0, 1.0 );
            lin += 4.5 * material * float3(0.8,0.4,0.45) * occ + 2.0 * material * float3(0.5,0.41,0.39) * bou * occ;
        }
        // sss
        {
            float3 lig = normalize(float3(0.2,1.0,0.05));
            float3 hal  = normalize(lig - rd);
            float dif = clamp(dot(normal, hal), 0.0, 1.0);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
            float3 fSch = f0 + (float3(1) - f0)*pow(fre, 5.0);
            lin += 3.5 * (1.0 - K.x) * fre * fre * (0.2 + 0.8 * dif * occ) * material;
        }
        
        // sun
        // {
        //     float dif = clamp(dot(nor, hal), 0.0, 1.0);
        //     dif *= shadow;
        //     float spe = K.x * pow( clamp(dot(nor, hal), 0.0, 1.0), K.y );
        //     spe *= dif;
        //     spe *= 0.04+0.96*pow(clamp(1.0-dot(hal,lig),0.0,1.0),5.0); // fresnel
        //     lin += material * (3.20) * dif * float3(2.1,1.1,0.6);
        //     lin += material * (3.) * spe * float3(2.1,1.1,0.6);
        // }
         // sky / reflections
         {
//             float dif = occ * sqrt(clamp(0.5 + 0.5 * normal.y, 0.0, 1.0));
//             float3 skyCol = skyColor(pos, ref, float3(8,-1,-8), time);
//             lin += material * 0.3 * dif * (0.35 * clamp(skyCol, 0.0, 0.9));
//
//             float spe = smoothstep( -0.2, 0.2, ref.y );
//             spe *= dif;
//             float fre = clamp(1.0 - dot(float3(8,-1,-8), normal), 0.0, 1.0);
//             spe *= occ * occ * K.x * 0.04 + 0.96 * pow( fre, 5.0 ) * rou;
////             spe *= shadow;
//             lin += 0.5 * spe * skyCol;
         }
        // // ground / bounce light
        // {
        //     lin += 2.0 * material * float3(0.5,0.41,0.39) * bou * occ;
        // }
        // // back
        // {
        //     float dif = clamp( dot( nor, normalize(float3(-0.5,0.0,-0.75))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
        //           dif *= occ;
        //     lin += 2.0*material*0.55*dif*float3(0.35,0.25,0.35);
        // }
        // // sss
        // {
        //     float dif = clamp(dot(norx, hal), 0.0, 1.0);
        //     lin += 3.5 * fre * fre * (0.2 + 0.8 * dif * occ * shadow) * material;
        // }

//        float fade = smoothstep(-16.0, -17.0, pos.z);
        float fade = 0.0;
        col = (1.0 - fade) * ((0.05 * material) + (0.95 * lin)) + fade*float3(0.92);
        if (secLin.x > 0.01) {
            col = secLin;
        }
    }
    
    return float3( clamp(col, 0.0, 1.0) );
}

kernel void compute(texture2d<float, access::write> output [[texture(0)]],
                    array<texture2d<float, access::sample>, 20> ground [[texture(1)]],
                    sampler defaultSampler [[sampler(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    constant Uniforms &uniforms [[buffer(0)]])
{
    float3 ro = float3( 0.0, 4.5, 1.5);
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
    
    float3 color = render( ro, rd, rdx, rdy, uniforms.time, ground, defaultSampler );

    color = pow(color, float3(0.4545));

    total += color;

    output.write(float4(total, 1.0), gid);
}
