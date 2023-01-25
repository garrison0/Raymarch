
import Cocoa

class ViewController: NSViewController {

    var renderer: Renderer!

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }

        guard let newRenderer = Renderer(device: defaultDevice) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer
        
        let dataURLString = NSString(string: "~/").expandingTildeInPath
        let movieURL = URL(fileURLWithPath: "movieone.m4v", relativeTo: URL(fileURLWithPath: dataURLString))
        try? FileManager.default.removeItem(at: movieURL)
        
        // 2560 / 1440
        // 1920 x 1080
        // 1280 x 720
        renderer.renderMovie(size: CGSize(width: 1280, height: 720), duration: 5, url: movieURL) {
            DispatchQueue.main.async {
                NSWorkspace.shared.activateFileViewerSelecting([movieURL.absoluteURL])
//                self.view.window?.close()
            }
        }
    }
}
