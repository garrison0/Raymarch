import Foundation

let width = 800
let height = 400
let t0 =  ProcessInfo.processInfo.systemUptime
let image = imageFromPixels(width: width, height: height)
let t1 =  ProcessInfo.processInfo.systemUptime
t1-t0
image
