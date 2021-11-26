import Flutter
import UIKit
import ImageIO
import FirebaseMLModelDownloader
import TensorFlowLite
import CoreImage
import Accelerate


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
         let batchSize = 1
         let inputChannels = 3
         let inputWidth = 224
         let inputHeight = 224

         
         let arguments = call.arguments as! NSDictionary
         let modelPath = arguments["modelPath"] as! String
         let imgBytes = arguments["imgFileBytes"] as! FlutterStandardTypedData
         
         var uiImage = UIImage(data: imgBytes.data)
//         let image = uiImage?.cgImage
         
         let scaledSize = CGSize(width: inputWidth, height: inputHeight)
         if #available(iOS 15.0, *) {
             uiImage = uiImage?.preparingThumbnail(of: scaledSize)
         } else {
             // Fallback on earlier versions
         }
         
         let pixelBuffer = uiImage?.convertToBuffer()
         
         
         var option = Interpreter.Options()
         option.threadCount = 1
//         option.isXNNPackEnabled = true
         
         let interpreter = try? Interpreter(modelPath: modelPath, options: option)
         try? interpreter?.allocateTensors()
         
         let outputTensor: Tensor
         do {
           // Allocate memory for the model's input `Tensor`s.
           try interpreter?.allocateTensors()
           let inputTensor = try interpreter?.input(at: 0)

           // Remove the alpha component from the image buffer to get the RGB data.
           guard let rgbData = rgbDataFromBuffer(
             pixelBuffer!,
             byteCount: batchSize * inputWidth * inputHeight * inputChannels,
             isModelQuantized: inputTensor?.dataType == .uInt8
           ) else {
             print("Failed to convert the image buffer to RGB data.")
             return
           }

           // Copy the RGB data to the input `Tensor`.
           try interpreter?.copy(rgbData, toInputAt: 0)

           // Run inference by invoking the `Interpreter`.
           try interpreter?.invoke()

           // Get the output `Tensor` to process the inference results.
           outputTensor = try interpreter!.output(at: 0)
//           print(outputTensor.shape)
             
             
           
             let results: [Float]
             switch outputTensor.dataType {
             case .uInt8:
               guard let quantization = outputTensor.quantizationParameters else {
                 print("No results returned because the quantization values for the output tensor are nil.")
                 return
               }
               let quantizedResults = [UInt8](outputTensor.data)
               results = quantizedResults.map {
                 quantization.scale * Float(Int($0) - quantization.zeroPoint)
               }
             case .float32:
               results = [Float32](unsafeData: outputTensor.data) ?? []
             default:
               print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
               return
             }

//             print(results)
             result(results)

             
         } catch let error {
           print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
           return
         }

         
         
     }
    
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
      ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
          CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
          return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
          print("Error: out of memory")
          return nil
        }
        
        defer {
            free(destinationData)
        }

        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)

        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)

        switch (pixelBufferFormat) {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            // Unknown pixel format.
            return nil
        }

        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }

        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append(Float(bytes[i]) / 255.0)
        }
        return Data(copyingBufferOf: floats)
      }
}


extension UIImage {
        
    func convertToBuffer() -> CVPixelBuffer? {
        
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, Int(self.size.width),
            Int(self.size.height),
            kCVPixelFormatType_32ARGB,
            attributes,
            &pixelBuffer)
        
        guard (status == kCVReturnSuccess) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        let context = CGContext(
            data: pixelData,
            width: Int(self.size.width),
            height: Int(self.size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: self.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context!)
        self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
    }
    
}


extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
