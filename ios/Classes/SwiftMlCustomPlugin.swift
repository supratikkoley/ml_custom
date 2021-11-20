import Flutter
import UIKit
import ImageIO
import FirebaseMLModelDownloader
import TensorFlowLite

public class SwiftMlCustomPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "ml_custom", binaryMessenger: registrar.messenger())
    let instance = SwiftMlCustomPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

   public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
       if(call.method == "getModel") {
           getModel(call: call, result: result)
       } else if (call.method == "isModelDownloaded") {
           isModelDownloaded(call: call, result: result)
       } else if (call.method == "runModelOnImage") {
           runModelOnImage(call: call, result: result)
       }
   }

    public func getModel(call: FlutterMethodCall, result: @escaping FlutterResult) {
        let arguments = call.arguments as! NSDictionary
        let modelName = arguments["modelName"] as! String
        
        let conditions = ModelDownloadConditions(allowsCellularAccess: false)
        
        var modelPath: String = "nil";
        
        ModelDownloader.modelDownloader().getModel(name: modelName, downloadType: .latestModel,
        conditions: conditions)
        { res in
            switch (res) {
            case .success(let customModel):
                do {
                    // Download complete. Depending on your app, you could enable the ML
                    // feature, or switch from the local model to the remote model, etc.

                    // The CustomModel object contains the local path of the model file,
                    // which you can use to instantiate a TensorFlow Lite interpreter.
//                    let interpreter = try Interpreter(modelPath: customModel.path)
                    modelPath =  customModel.path
                    result(modelPath)
            
                }
            case .failure(let error):
                modelPath = error.localizedDescription
                result(modelPath)
            }
        }
    }
    
    public func isModelDownloaded(call: FlutterMethodCall, result: @escaping FlutterResult) {
        let arguments = call.arguments as! NSDictionary
        let modelName = arguments["modelName"] as! String
//        var isDownloaded = false
        ModelDownloader.modelDownloader().listDownloadedModels(){
            res in
            switch (res){
            case .success(let models):
                do {
                    var f = 0
                    for model in models{
                        if(model.name == modelName){
                            result(true)
                            f += 1
                            break
                        }
                    }
                    if(f == 0){
                        result(false)
                    }
                    
                }
            case .failure(let error):
                result(error)
            }
        }
    }
    
     public func runModelOnImage(call: FlutterMethodCall, result: @escaping FlutterResult) {
         
         
         let arguments = call.arguments as! NSDictionary
         let modelPath = arguments["modelPath"] as! String
         let imgBytes = arguments["imgFileBytes"] as! FlutterStandardTypedData
         
         let uiImage = UIImage(data: imgBytes.data, scale: 1.0)
         
         
         
         let inputData = uiImage?.jpegData(compressionQuality: 1.0)
         
         
         var option = Interpreter.Options()
         option.threadCount = 1
//         option.isXNNPackEnabled = true
         
         let interpreter = try? Interpreter(modelPath: modelPath, options: option)
         
         try? interpreter?.allocateTensors()
         let _ = try? interpreter?.copy(inputData!, toInputAt: 0)
         try? interpreter?.invoke()
         
         let output = try? interpreter?.output(at: 0)
         
         print(output?.data ?? "")
         
         let outputDim = output?.shape.dimensions
  
         let probabilities =
         UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputDim?[1] ?? 0)
        
         let _ = output?.data.copyBytes(to: probabilities)
         
         var outputArray: Array = Array(repeating: Float32(0), count: 0)
        
         
         for prob in probabilities {
             outputArray.append(prob)
         }
         
         result(outputArray)
         
     }
    
    func buffer(from image: UIImage) -> CVPixelBuffer? {
      let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
      var pixelBuffer : CVPixelBuffer?
      let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
      guard (status == kCVReturnSuccess) else {
        return nil
      }

      CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
      let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

      let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
      let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

      context?.translateBy(x: 0, y: image.size.height)
      context?.scaleBy(x: 1.0, y: -1.0)

      UIGraphicsPushContext(context!)
      image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
      UIGraphicsPopContext()
      CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

      return pixelBuffer
    }
    
    
    
}


