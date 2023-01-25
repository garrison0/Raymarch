
import Metal
import MetalKit
import AVFoundation
import simd

let maxBuffersInFlight = 3

class Renderer: NSObject {

    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBuffer: MTLBuffer
    var time: Float = 0
    var cubemap: MTLTexture?
    var textures: [MTLTexture]?
    var cps: MTLComputePipelineState?
    var taaCps: MTLComputePipelineState?
    var smaaCps: MTLComputePipelineState?
    var cmpCps: MTLComputePipelineState?
    var cldCps: MTLComputePipelineState?
    var sampler: MTLSamplerState

    init?(device: MTLDevice) {
        self.device = device
        self.commandQueue = self.device.makeCommandQueue()!

        uniformBuffer = Renderer.createBuffer(device)
        
        do {
            try cps = Renderer.registerWorldShader(device)
        } catch let error {
            print("\(error)")
        }
        
        do {
            try taaCps = Renderer.registerTAAShader(device)
        } catch let error {
            print("\(error)")
        }
        
        do {
            try cmpCps = Renderer.registerCompositeShader(device)
        } catch let error {
            print("\(error)")
        }
        
        do {
            try cldCps = Renderer.registerCloudShader(device)
        } catch let error {
            print("\(error)")
        }
        
        do {
            try smaaCps = Renderer.registerSMAAShader(device)
        } catch let error {
            print("\(error)")
        }
        
        do {
            try textures = Renderer.setUpTextures(device)
        } catch let error {
            print("\(error)")
        }
        
        do {
            try cubemap = Renderer.setUpCubeMap(device)
        } catch let error {
            print("\(error)")
        }
        
        sampler = Renderer.buildSamplerState(device)

        super.init()
    }

    class func registerWorldShader(_ device: MTLDevice) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "compute")
        return try device.makeComputePipelineState(function: kernel!)
    }
    
    class func registerCloudShader(_ device: MTLDevice) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "computeClouds")
        return try device.makeComputePipelineState(function: kernel!)
    }
    
    class func registerCompositeShader(_ device: MTLDevice) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "compositeTextures")
        return try device.makeComputePipelineState(function: kernel!)
    }
    
    class func registerTAAShader(_ device: MTLDevice) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "taa")
        return try device.makeComputePipelineState(function: kernel!)
    }
    
    class func registerSMAAShader(_ device: MTLDevice) throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()!
        let kernel = library.makeFunction(name: "filmicSMAA")
        return try device.makeComputePipelineState(function: kernel!)
    }
    
    class func setUpCubeMap(_ device: MTLDevice) throws -> MTLTexture? {
        let textureLoader = MTKTextureLoader.init(device: device);
        let cubeSize = 2048;
        let textureCubeDescriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: MTLPixelFormat.rgba8Unorm, size: cubeSize, mipmapped: false)
        let cubemap = device.makeTexture(descriptor: textureCubeDescriptor)

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
                cubemap?.replace(region: region, mipmapLevel: 0, slice: slice, withBytes: data, bytesPerRow: bytesPerRow, bytesPerImage: bytesPerRow * cubeSize);
            }
            slice += 1
        }
        return cubemap;
    }
    
    class func setUpTextures(_ device: MTLDevice) throws -> [MTLTexture]? {
        let textures: [MTLTexture]
        let textureLoader = MTKTextureLoader(device: device)
        if let paths = Bundle.main.urls(forResourcesWithExtension: "jpg", subdirectory: nil) {
            let sorted = paths.sorted(by: { $0.lastPathComponent.lowercased() <                         $1.lastPathComponent.lowercased() })
            let errorPtr: NSErrorPointer = nil
            let options: [MTKTextureLoader.Option : Any] = [.generateMipmaps : true, .allocateMipmaps: true, .SRGB : true ]
            textures = textureLoader.newTextures(URLs: sorted, options: options, error: errorPtr)
            if let error = errorPtr?.pointee {
                throw(error)
            }
            return textures;
        }
        throw NSError(domain: "No resources found.", code: 0)
    }
    
    class func createBuffer(_ device: MTLDevice) -> MTLBuffer {
        var uniformBufferInit = Uniforms(time: 0.0, resolution: SIMD2<Float>(0.0, 0.0))
        return device.makeBuffer(bytes: &uniformBufferInit, length: MemoryLayout<Uniforms>.stride, options: [])!
    }
    
    class func buildSamplerState(_ device: MTLDevice) -> MTLSamplerState {
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.normalizedCoordinates = true
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.rAddressMode = .repeat
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .repeat
        return device.makeSamplerState(descriptor: samplerDescriptor)!
    }

    private func updateGameState(forTime time: Float) {
        self.time = time
    }
    
    func renderMovie(size: CGSize, duration: TimeInterval, url: URL, completion: @escaping () -> Void) {
        
        let recorder = VideoRecorder(outputURL: url, size: size)!
        
        let lowResTextureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                               width: 1440,
                                                                               height: 810,
                                                                               mipmapped: false);
        lowResTextureDescriptor.usage = [ .shaderWrite, .shaderRead ]
        lowResTextureDescriptor.storageMode = .managed
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                         width: Int(size.width),
                                                                         height: Int(size.height),
                                                                         mipmapped: false)
        textureDescriptor.usage = [ .shaderWrite, .shaderRead ]
        textureDescriptor.storageMode = .managed
        
        let buffers = [device.makeTexture(descriptor: textureDescriptor)!,
                       device.makeTexture(descriptor: textureDescriptor)!,
                       device.makeTexture(descriptor: textureDescriptor)!,
                       device.makeTexture(descriptor: textureDescriptor)!,
                       device.makeTexture(descriptor: textureDescriptor)!];
        let framerate = 10.0;
        let frameDelta = 1 / framerate
        
        recorder.startRecording()
        
        for t in stride(from: 0, through: duration, by: frameDelta) {
            self.draw(in: buffers, time: t) { (texture) in
                recorder.writeFrame(forTexture: texture, time: t)
            }
        }
        
        recorder.endRecording {
            completion()
        }
    }

    func draw(in renderTextures: [MTLTexture], time: TimeInterval, completion: @escaping (MTLTexture) -> Void) {
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        let ptr = uniformBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
            ptr.pointee.time = Float(time)
            ptr.pointee.resolution = SIMD2<Float>(Float(renderTextures[1].width), Float(renderTextures[1].height));
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_) in
                semaphore.signal()
            }
            
            commandBuffer.addCompletedHandler { (_) in
                completion(renderTextures[4]) //set to 2
            }
            
            self.updateGameState(forTime: Float(time))

            // render world, SDF objects (into buffer 1)
            let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            commandEncoder.setComputePipelineState(cps!)
            commandEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)
            commandEncoder.setTexture(renderTextures[1], index: 0)
            commandEncoder.setTexture(cubemap, index: 1)
            commandEncoder.setTextures(textures!, range: 2..<15)
            commandEncoder.setSamplerState(sampler, index: 0)
            let threadGroupCount = MTLSizeMake(8, 8, 1)
            var threadGroups = MTLSizeMake(renderTextures[1].width / threadGroupCount.width, renderTextures[1].height / threadGroupCount.height, 1)
            commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
            commandEncoder.endEncoding()
            
            // render volumetric clouds (into buffer 0)
            let cldCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
            cldCommandEncoder.setComputePipelineState(cldCps!)

            cldCommandEncoder.setTexture(renderTextures[0], index: 0)
            cldCommandEncoder.setTexture(renderTextures[1], index: 1)
            cldCommandEncoder.setSamplerState(sampler, index: 0)
            cldCommandEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)

            threadGroups = MTLSizeMake(renderTextures[0].width / threadGroupCount.width, renderTextures[0].height / threadGroupCount.height, 1)
            cldCommandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)

            cldCommandEncoder.endEncoding()
            
            // composite current frame (into buffer 2)
            let cmpCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
            cmpCommandEncoder.setComputePipelineState(cmpCps!)

            cmpCommandEncoder.setTexture(renderTextures[0], index: 0)
            cmpCommandEncoder.setTexture(renderTextures[1], index: 1)
            cmpCommandEncoder.setTexture(renderTextures[2], index: 2)
            cmpCommandEncoder.setSamplerState(sampler, index: 0)
            cmpCommandEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)

            threadGroups = MTLSizeMake(renderTextures[2].width / threadGroupCount.width, renderTextures[2].height / threadGroupCount.height, 1)
            cmpCommandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)

            cmpCommandEncoder.endEncoding()
            
            // TAA post processing pass (current frame = buffer 2, previous frame = 3)
            let taaCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
            taaCommandEncoder.setComputePipelineState(taaCps!)

            taaCommandEncoder.setTexture(renderTextures[2], index: 0)
            taaCommandEncoder.setTexture(renderTextures[3], index: 1)
            taaCommandEncoder.setTexture(renderTextures[3], index: 2)
            taaCommandEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)

            taaCommandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)

            taaCommandEncoder.endEncoding()
            
            // SMAA filmic step
            let smaaCommandEncoder = commandBuffer.makeComputeCommandEncoder()!
            smaaCommandEncoder.setComputePipelineState(smaaCps!)

            smaaCommandEncoder.setTexture(renderTextures[3], index: 0)
            smaaCommandEncoder.setTexture(renderTextures[4], index: 1)
            smaaCommandEncoder.setTexture(renderTextures[4], index: 2)
            smaaCommandEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)
            smaaCommandEncoder.setSamplerState(sampler, index: 0)

            smaaCommandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)

            smaaCommandEncoder.endEncoding()
//            let copybackEncoder = commandBuffer.makeBlitCommandEncoder()!
//            copybackEncoder.synchronize(resource: texture)
//            copybackEncoder.endEncoding()
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
}
