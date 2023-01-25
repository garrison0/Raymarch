import simd

protocol Hitable {
    func hit(r: ray, _ tmin: Float, _ tmax: Float) -> Hit_record?
}

struct Hit_record {
    var t: Float
    var p: SIMD3<Float>
    var normal: SIMD3<Float>
    var mat_ptr: material
}

struct Hitable_list: Hitable  {
    var list: [Hitable]
    
    func hit(r: ray, _ tmin: Float, _ tmax: Float) -> Hit_record? {
        var hit_anything: Hit_record?
        for item in list {
          if let aHit = item.hit(r: r, tmin, hit_anything?.t ?? tmax) {
                hit_anything = aHit
            }
        }
        return hit_anything
    }
}

struct Sphere: Hitable  {
    var center: SIMD3<Float>
    var radius: Float
    var mat: material
    
    init(c: SIMD3<Float>, r: Float, m: material) {
        center = c
        radius = r
        mat = m
    }
    
    func hit(r: ray, _ tmin: Float, _ tmax: Float) -> Hit_record? {
        let oc = r.origin - center
        let a = dot(r.direction, r.direction)
        let b = dot(oc, r.direction)
        let c = dot(oc, oc) - radius * radius
        let discriminant = b * b - a * c
        if discriminant > 0 {
            var t = (-b - sqrt(discriminant) ) / a
            if t < tmin {
                t = (-b + sqrt(discriminant) ) / a
            }
            if tmin < t && t < tmax {
                let point = r.point_at_parameter(t: t)
                let normal = (point - center) / SIMD3<Float>(repeating: radius)
                return Hit_record(t: t, p: point, normal: normal, mat_ptr: mat)
            }
        }
        return nil
    }
}

func random_in_unit_sphere() -> SIMD3<Float> {
    var p = SIMD3<Float>()
    repeat {
        p = 2.0 * SIMD3<Float>(x: Float(drand48()), y: Float(drand48()), z: Float(drand48())) - SIMD3<Float>(x: 1, y: 1, z: 1)
    } while dot(p, p) >= 1.0
    return p
}
