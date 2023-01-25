import simd

protocol material {
    func scatter(ray_in: ray, _ rec: Hit_record, _ attenuation: inout SIMD3<Float>, _ scattered: inout ray) -> Bool
}
 
class lambertian: material {
    var albedo: SIMD3<Float>
    init(a: SIMD3<Float>) {
        albedo = a
    }
    func scatter(ray_in: ray, _ rec: Hit_record, _ attenuation: inout SIMD3<Float>, _ scattered: inout ray) -> Bool {
        let target = rec.p + rec.normal + random_in_unit_sphere()
        scattered = ray(origin: rec.p, direction: target - rec.p)
        attenuation = albedo
        return true
    }
}

class metal: material {
    var albedo: SIMD3<Float>
    var fuzz: Float
    init(a: SIMD3<Float>, f: Float) {
        albedo = a
        if f < 1 {
            fuzz = f
        } else {
            fuzz = 1
        }
    }
    func scatter(ray_in: ray, _ rec: Hit_record, _ attenuation: inout SIMD3<Float>, _ scattered: inout ray) -> Bool {
        let reflected = reflect(normalize(ray_in.direction), n: rec.normal)
        scattered = ray(origin: rec.p, direction: reflected + fuzz * random_in_unit_sphere())
        attenuation = albedo
        return dot(scattered.direction, rec.normal) > 0
    }
}
