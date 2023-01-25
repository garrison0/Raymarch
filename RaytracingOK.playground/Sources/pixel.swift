import CoreImage

public struct Pixel {
    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8
    init(red: UInt8, green: UInt8, blue: UInt8) {
        r = red
        g = green
        b = blue
        a = 255
    }
}

public func imageFromPixels(width: Int, height: Int) -> CIImage {
    var pixel = Pixel(red: 0, green: 0, blue: 0)
    var pixels = Array(repeating: pixel, count: width * height)

    let world = hitable_list()
    
    var object = sphere(c: SIMD3<Float>(x: 0, y: -100.5, z: -1), r: 100, m: lambertian(a: SIMD3<Float>(x: 0, y: 0.7, z: 0.3)))
    world.add(h: object)
    object = sphere(c: SIMD3<Float>(x: 1, y: 0, z: -1.1), r: 0.5, m: metal(a: SIMD3<Float>(x: 0.8, y: 0.6, z: 0.2), f: 0.7))
    world.add(h: object)
    object = sphere(c: SIMD3<Float>(x: -1, y: 0, z: -1.1), r: 0.5, m: metal(a: SIMD3<Float>(x: 0.8, y: 0.8, z: 0.8), f: 0.1))
    world.add(h: object)
    object = sphere(c: SIMD3<Float>(x: 0, y: 0, z: -1), r: 0.5, m: lambertian(a: SIMD3<Float>(x: 0.3, y: 0, z: 0)))
    world.add(h: object)
    
    let cam = camera()
    for i in 0..<width {
        for j in 0..<height {
            let ns = 10
            var col = SIMD3<Float>()
            for _ in 0..<ns {
                let u = (Float(i) + Float(drand48())) / Float(width)
                let v = (Float(j) + Float(drand48())) / Float(height)
                let r = cam.get_ray(u: u, v)
                col += color(r: r, world)
            }
            col /= SIMD3<Float>(repeating: Float(ns));
            pixel = Pixel(red: UInt8(col.x * 255), green: UInt8(col.y * 255), blue: UInt8(col.z * 255))
            pixels[i + j * width] = pixel
        }
    }
    
    let bitsPerComponent = 8
    let bitsPerPixel = 32
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    let providerRef = CGDataProvider(data: NSData(bytes: pixels, length: pixels.count * MemoryLayout<Pixel>.size))
    let image = CGImage(width: width, height: height, bitsPerComponent: bitsPerComponent, bitsPerPixel: bitsPerPixel, bytesPerRow: width * MemoryLayout<Pixel>.size, space: rgbColorSpace, bitmapInfo: bitmapInfo, provider: providerRef!, decode: nil, shouldInterpolate: true, intent: CGColorRenderingIntent.defaultIntent)
    return CIImage(cgImage: image!)
}
