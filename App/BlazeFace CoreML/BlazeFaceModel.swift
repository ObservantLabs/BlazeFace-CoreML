//
//  BlazeFaceModel.swift
//  BlazeFace CoreML
//
//  Created by Vidur Satija on 07/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import Foundation
import Vision
import CoreML
import Accelerate
import CoreGraphics
import CoreImage
import VideoToolbox

// takes an image and presents itself as an MLFeatureProvidre with a 128x128 image
class BlazeFaceInput: MLFeatureProvider {
    private static let imageFeatureName = "image"

    var imageFeature: CGImage

    var featureNames: Set<String> {
        return [BlazeFaceInput.imageFeatureName]
    }

    init(image: CGImage) {
        imageFeature = image
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == BlazeFaceInput.imageFeatureName else {
            return nil
        }

        let options: [MLFeatureValue.ImageOption: Any] = [
            .cropAndScale: VNImageCropAndScaleOption.scaleFit.rawValue
        ]

        return try? MLFeatureValue(cgImage: imageFeature,
                                   pixelsWide: 128,
                                   pixelsHigh: 128,
                                   pixelFormatType: imageFeature.pixelFormatInfo.rawValue,
                                   options: options)
    }
}

public func IOU(_ a: SIMD16<Float32>, _ b: SIMD16<Float32>) -> Float {
    let areaA = (a[3]-a[1]) * (a[2]-a[0])
    if areaA <= 0 { return 0 }

    let areaB = (b[3]-b[1]) * (b[2]-b[0])
    if areaB <= 0 { return 0 }

    let intersectionMinX = max(a[1], b[1])
    let intersectionMinY = max(a[0], b[0])
    let intersectionMaxX = min(a[3], b[3])
    let intersectionMaxY = min(a[2], b[2])
    let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                            max(intersectionMaxX - intersectionMinX, 0)
    return Float(intersectionArea / (areaA + areaB - intersectionArea))
}


class BlazeFaceModel {
    var model: MLModel?
    let minConfidence: Float32 = 0.75
    let nmsThresh = 0.3
    
    init() {
      // BlazeFacceScaled is the auto-generated model
        self.model = BlazeFaceScaled().model
    }

  // called from av frame loop
    func predict(for buffer: CVPixelBuffer) -> (landmarks: [SIMD16<Double>], confidence: [Float32]) {

        var imageFeature: CGImage?
        VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &imageFeature)
        let imgH = Float32(imageFeature!.height)
        let imgW = Float32(imageFeature!.width)
        
        let hScale = max(imgH, imgW) / imgH
        let wScale = max(imgH, imgW) / imgW
        // let wB = wScale - 1
        
        let x = BlazeFaceInput(image: imageFeature!)
        guard let points = try? self.model!.prediction(from: x) else {
          print("Failed to return any points")
            return ([], [])
        }

      print("output shape: \(points)")
      /*
       The __raw BlazeFaceScaled MLModel__ returns a dicationary with two keys, "1477" and "1011", corresponding
       to two multiarrays, one with dimensions (1,896,16) and the other (1,896,1)

       WIth the following VALUE LAYOUT:


       self.faceView.boundingBox.append(CGRect(x: fL[0], y: fL[8], width: fL[1]-fL[0], height: fL[9]-fL[8]))
       self.faceView.rightEye.append(CGPoint(x: fL[2], y: fL[10]))
       self.faceView.leftEye.append(CGPoint(x: fL[3], y: fL[11]))
       self.faceView.nose.append(CGPoint(x: fL[4], y: fL[12]))
       self.faceView.mouth.append(CGPoint(x: fL[5], y: fL[13]))
       self.faceView.rightEar.append(CGPoint(x: fL[6], y: fL[14]))
       self.faceView.leftEar.append(CGPoint(x: fL[7], y: fL[15]))

       [
       0: BB. xmin
       1: BB. xmax
       2: rightEye,x
       3: leftEye.x
       4: nose.x
       5: mouth.x
       6: rightEar.x
       7: leftEar.x
       8: BB.ymin
       9: BB ymax
       10: rightEye.y
       11: leftEye.y
       12: nose.y
       13: mouth.y
       14: rightEar.y
       15: leftEar.y
       ]  = 1st block of 16 float32

       [
       ....
       ] = 2nd block of 16 float32


       */

        let rPointsMLArray = points.featureValue(for: "1477")?.multiArrayValue
      assert(rPointsMLArray!.count == 896 * 16, "Unexpected total number of elements in tensor named 1477")
        let rPoints = rPointsMLArray?.dataPointer.bindMemory(to: SIMD16<Float32>.self, capacity: rPointsMLArray!.count/16) // 896 x 8 x 2 -> 2 bounding box + 6 keypoints
      // rArray, an array of 896 elements, where every element is a SIMD16 (which is an array of 16 elements)
      let rArray = [SIMD16<Float32>](UnsafeBufferPointer(start: rPoints, count: rPointsMLArray!.count/16))


        let cMLArray = points.featureValue(for: "1011")?.multiArrayValue
      assert(cMLArray!.count == 896, "Unexpected total number of elements in the tensor named 1011")

        let c = cMLArray?.dataPointer.bindMemory(to: Float32.self, capacity: cMLArray!.count)
      // cArray, array of 896 elements, where every element is a float32 confidence value
        let cArray = [Float32](UnsafeBufferPointer(start: c, count: cMLArray!.count))
        
        // Apply custom NMS


        var cIndices = cArray.enumerated().filter({ $0.element >= self.minConfidence }).map({ $0.offset })
        cIndices.sort(by: { cArray[$0] > cArray[$1] })

      // assert: cIndicies is an array of the indices of all elements whose confidence value is above minConfidence, sorted in order of decreasing confidence



        var retRArray = Array<SIMD16<Double>>()
        var retCArray = Array<Float32>()


        while cIndices.count > 0 {
            var overlapRs = Array<SIMD16<Float32>>()
            var overlapCscore: Float32 = 0.0
            var nonOverlapI = Array<Int>()

          // for every high-confidence recognigition, with the index i ...
            for i in 0..<cIndices.count {

              // .. check the intersection-over-union with the i==0 recognition
              let iiou:Float = IOU(rArray[cIndices[0]], rArray[cIndices[i]])


              // "remove overlapping ones and average them out" ???

              // if the IoU is high ...
              if iiou >= 0.3 {

                // .. then
                overlapRs.append(cArray[cIndices[i]]*rArray[cIndices[i]])
                overlapCscore += cArray[cIndices[i]]
              } else {
                nonOverlapI.append(cIndices[i])
              }
            }
            cIndices = nonOverlapI
            let averageR = overlapRs.reduce(SIMD16<Float32>(repeating: 0), +) / overlapCscore
            var betterR = SIMD16<Double>(repeating: 0)
            for i in 0..<16 {
                if i % 2 == 0 {
                    betterR[i/2] = Double(averageR[i]*wScale - (wScale-1)/2.0) // all Xs
                } else {
                    betterR[8+i/2] = Double(averageR[i]*hScale - (hScale-1)/2.0) // all Ys
                }
            }
            retRArray.append(betterR)
            retCArray.append(overlapCscore / Float32(overlapRs.count))
        }
        return (retRArray, retCArray)
    }
}
