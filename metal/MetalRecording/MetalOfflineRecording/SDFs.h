
#ifndef SDFs_h
#define SDFs_h

float2 opU( float2 d1, float2 d2 );

float2 opSmoothU( float2 d1, float2 d2, float k);

float opSmoothSubtraction( float d1, float d2, float k );

float opIntersection( float d1, float d2 );

float sdfSphere(float3 p, float r);

float sdTorus( float3 p, float2 t );

float sdBox( float3 p, float3 b );

float sdRoundBox( float3 p, float3 b, float r );

float sdHexPrism( float3 p, float2 h );

float sdbEllipsoid( float3 p, float3 r );

float2 sdCapsule( float3 p, float3 a, float3 b );

float sdCappedCone(float3 p, float3 a, float3 b, float ra, float rb);

float sdCappedCylinderVertical( float3 p, float h, float r );

float sdCappedCylinder(float3 p, float3 a, float3 b, float r);

float sdTriangleIsosceles( float2 p, float2 q );

float sdExtrudedTriangle(float3 p, float2 q, float h);

float sdOctogon( float2 p, float r );

float sdExtrudedOctogon(float3 p, float r, float h);

float sdCappedTorus( float3 p, float2 sc, float ra, float rb);

#endif /* SDFs_h */
