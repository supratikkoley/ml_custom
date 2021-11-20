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
         
         let uiImage = UIImage(data: imgBytes.data)
//         let image = uiImage?.cgImage
         
         
         
//         guard let context = CGContext(
//           data: nil,
//           width: image!.width, height: image!.height,
//           bitsPerComponent: 8, bytesPerRow: image!.width * 4,
//           space: CGColorSpaceCreateDeviceRGB(),
//           bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
//         ) else {
//           return
//         }
//
//         context.draw(image!, in: CGRect(x: 0, y: 0, width: image!.width, height: image!.height))
//         guard let imageData = context.data else { return }

         let inputData = uiImage?.jpegData(compressionQuality: 1.0)
         
//        for row in 0 ..< 224 {
//            for col in 0 ..< 224 {
//                let offset = 4 * (row * context.width + col)
//                // (Ignore offset 0, the unused alpha channel)
//                let red = imageData.load(fromByteOffset: offset+1, as: UInt8.self)
//                let green = imageData.load(fromByteOffset: offset+2, as: UInt8.self)
//                let blue = imageData.load(fromByteOffset: offset+3, as: UInt8.self)
//
//                // Normalize channel values to [0.0, 1.0]. This requirement varies
//                // by model. For example, some models might require values to be
//                // normalized to the range [-1.0, 1.0] instead, and others might
//                // require fixed-point values or the original bytes.
//                var normalizedRed = Float32(red) / 255.0
//                var normalizedGreen = Float32(green) / 255.0
//                var normalizedBlue = Float32(blue) / 255.0
//
//                // Append normalized values to Data object in RGB order.
//                let elementSize = MemoryLayout.size(ofValue: normalizedRed)
//                var bytes = [UInt8](repeating: 0, count: elementSize)
//                memcpy(&bytes, &normalizedRed, elementSize)
//                inputData.append(&bytes, count: elementSize)
//                memcpy(&bytes, &normalizedGreen, elementSize)
//                inputData.append(&bytes, count: elementSize)
//                memcpy(&bytes, &normalizedBlue, elementSize)
//                inputData.append(&bytes, count: elementSize)
//            }
//       }
//
         NSLog(inputData?.base64EncodedString() ?? "")
         
         let interpreter = try? Interpreter(modelPath: modelPath)
         
         NSLog("TFLite Desc: " + interpreter.debugDescription)
         
         try? interpreter?.allocateTensors()
         let _ = try? interpreter?.copy(inputData!, toInputAt: 0)
         try? interpreter?.invoke()
         
         let output = try? interpreter?.output(at: 0)
         
//         print(output)
         
         let outputDim = output?.shape.dimensions
         
//         print(outputDim?[0] ?? "NULL")
         
         
//
         let probabilities =
         UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputDim?[1] ?? 0)
        
//         print(probabilities.count)
         
         let _ = output?.data.copyBytes(to: probabilities)
         
         var outputArray: Array = Array(repeating: Float32(0), count: 0)
        
//         print(outputArray.count)
         for prob in probabilities {
             outputArray.append(prob)
         }
         
//         interpreter.
         
//          print(outputArray.count)
         
         result(outputArray)
         

         
     }
}


