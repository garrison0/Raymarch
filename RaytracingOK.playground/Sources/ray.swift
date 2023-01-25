import simd

struct ray {
    var origin: SIMD3<Float>
    var direction: SIMD3<Float>
    func point_at_parameter(t: Float) -> SIMD3<Float> {
        return origin + t * direction
    }
}

func color(r: ray, _ world: Hitable, _ depth: Int) -> SIMD3<Float> {
    var rec = Hit_record()
    if world.hit(r: r, 0.001, Float.infinity, &rec) {
        var scattered = r
        var attenuantion = SIMD3<Float>()
        if depth < 50 && rec.mat_ptr.scatter(ray_in: r, rec, &attenuantion, &scattered) {
            return attenuantion * color(r: scattered, world, depth + 1)
        } else {
            return SIMD3<Float>(x: 0, y: 0, z: 0)
        }
    } else {
        let unit_direction = normalize(r.direction)
        let t = 0.5 * (unit_direction.y + 1)
        return (1.0 - t) * SIMD3<Float>(x: 1, y: 1, z: 1) + t * SIMD3<Float>(x: 0.5, y: 0.7, z: 1.0)
    }
}

struct camera {
    let lower_left_corner: SIMD3<Float>
    let horizontal: SIMD3<Float>
    let vertical: SIMD3<Float>
    let origin: SIMD3<Float>
    init() {
        lower_left_corner = SIMD3<Float>(x: -2.0, y: 1.0, z: -1.0)
        horizontal = SIMD3<Float>(x: 4.0, y: 0, z: 0)
        vertical = SIMD3<Float>(x: 0, y: -2.0, z: 0)
        origin = SIMD3<Float>()
    }
    func get_ray(u: Float, _ v: Float) -> ray {
        return ray(origin: origin, direction: lower_left_corner + u * horizontal + v * vertical - origin);
    }
}
