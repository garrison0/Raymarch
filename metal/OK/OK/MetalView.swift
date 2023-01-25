
import Cocoa
import MetalKit
import CoreFoundation
import CoreImage
import Accelerate

class MetalView: MTKView {
    var commandQueue: MTLCommandQueue?
    var cps: MTLComputePipelineState!
    var cubemap: MTLTexture?
    var textures: [MTLTexture]?
    var uniformBuffer: MTLBuffer?
    var time: Float = 0
    var sampler: MTLSamplerState?
    var movieURL: URL?
    
    required init(coder: NSCoder) {
        super.init(coder: coder)
        self.preferredFramesPerSecond = 1;
        self.autoResizeDrawable = false;
        self.drawableSize = CGSize(width: 0.5*1280, height: 0.5*720)
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device!.makeCommandQueue()
        self.framebufferOnly = false

        createBuffer()
        setUpCubeMap()
        setUpTextures()
        registerShaders()
        buildSamplerState()
    }
    
    func buildSamplerState() {
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.normalizedCoordinates = true
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.rAddressMode = .repeat
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .repeat
        sampler = device!.makeSamplerState(descriptor: samplerDescriptor)!
    }

    func setUpTextures() {
        let textureLoader = MTKTextureLoader(device: device!)
        if let paths = Bundle.main.urls(forResourcesWithExtension: "jpg", subdirectory: nil) {
            let sorted = paths.sorted(by: { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() })
            let errorPtr: NSErrorPointer = nil
            let options: [MTKTextureLoader.Option : Any] = [.generateMipmaps : true, .SRGB : true ]
            textures = textureLoader.newTextures(URLs: sorted, options: options, error: errorPtr)
        }
    }
    func setUpCubeMap() {
        let textureLoader = MTKTextureLoader.init(device: self.device!);
        let cubeSize = 2048;
        let textureCubeDescriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: MTLPixelFormat.rgba8Unorm, size: cubeSize, mipmapped: false)
        self.cubemap = device?.makeTexture(descriptor: textureCubeDescriptor)

        let region = MTLRegionMake2D(0, 0, cubeSize, cubeSize);
        
        let paths = Bundle.main.urls(forResourcesWithExtension: "png", subdirectory: nil) ?? []
        let sorted = paths.sorted(by: { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() })
        var slice = 0;
        let bytesPerPixel = 4;
        let bytesPerRow = bytesPerPixel * cubeSize;
        for item in sorted {
            if let imageData = try? Data(contentsOf:item) {
                let texture = try? textureLoader.newTexture(data: imageData, options: nil)
                let data = UnsafeMutablePointer<UInt8>.allocate(capacity: cubeSize * cubeSize * 4)
                texture!.getBytes(data, bytesPerRow: 4 * cubeSize, from: region, mipmapLevel: 0)
                self.cubemap?.replace(region: region, mipmapLevel: 0, slice: slice, withBytes: data, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerRow * cubeSize);
            }
            slice += 1
        }
    }
    
    func createBuffer() {
        var uniformBufferInit = Uniforms(time: 0.0, resolution: SIMD2<Float>(0.0, 0.0))
        uniformBuffer = device!.makeBuffer(bytes: &uniformBufferInit, length: MemoryLayout<Uniforms>.stride, options: [])!
    }
    
    func registerShaders() {
        let library = device!.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "compute")
        do {
            cps = try device!.makeComputePipelineState(function: kernel!)
        } catch let error {
            self.printView("\(error)")
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        time += 1.0 / Float(self.preferredFramesPerSecond)

        let ptr = uniformBuffer!.contents().bindMemory(to: Uniforms.self, capacity: 1)
            ptr.pointee.time = time
            ptr.pointee.resolution = SIMD2<Float>(Float(drawableSize.width), Float(drawableSize.height));
        
        if let drawable = currentDrawable {
            let commandBuffer = commandQueue!.makeCommandBuffer()
            
            let commandEncoder = commandBuffer?.makeComputeCommandEncoder()
            commandEncoder?.setComputePipelineState(cps!)
            commandEncoder?.setBuffer(uniformBuffer, offset: 0, index: 0)
            commandEncoder?.setTexture(drawable.texture, index: 0)
            commandEncoder?.setTexture(cubemap, index: 1)
            commandEncoder?.setTextures(textures!, range: 2..<18)
            commandEncoder?.setSamplerState(sampler, index: 0)
            let threadGroupCount = MTLSizeMake(8, 8, 1)
            let threadGroups = MTLSizeMake(drawable.texture.width / threadGroupCount.width, drawable.texture.height / threadGroupCount.height, 1)
            commandEncoder?.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
            commandEncoder?.endEncoding()
            commandBuffer?.present(drawable)
            commandBuffer?.commit()
        }
    }
}
