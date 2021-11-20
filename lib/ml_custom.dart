import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class MlCustom {
  static const MethodChannel _channel = MethodChannel('ml_custom');

  static Future<File?> getModel(String modelName) async {
    final String? modelPath =
        await _channel.invokeMethod('getModel', {'modelName': modelName});
    if (modelPath != null) {
      // debugPrint("Model is downloaded.");
      return File(modelPath);
    }
    return null;
  }

  static Future<bool> isModelDownloaded(String modelName) async {
    final bool isDownloaded = await _channel
        .invokeMethod('isModelDownloaded', {"modelName": modelName});
    debugPrint("Model is downloaded: " + isDownloaded.toString());
    return isDownloaded;
  }

  static Future<List<Map<String, dynamic>>> runModelOnImage(
      {required String? imgPath,
      required String? modelPath,
      required String labelPath}) async {
    // debugPrint(imgPath);
    // debugPrint(modelPath);

    if (imgPath == null) throw "Image Path is null.";
    if (modelPath == null) throw "Model Path is null.";

    var imgBytes = await rootBundle.load(imgPath);
    var imgUint8List = imgBytes.buffer.asUint8List();

    var labelStr = await rootBundle.loadString(labelPath);

    var labels = labelStr.split('\n');


    final probs = await _channel.invokeMethod(
      'runModelOnImage',
      {
        'imgFileBytes': imgUint8List,
        "modelPath": modelPath,
        'labelPath': labelPath
      },
    ) as List?;

    List<Map<String, dynamic>> result = [];

    var _minLen = min(probs?.length ?? 0, labels.length);

    for (int i = 0; i < _minLen; i++) {
      result.add({'prob': probs?[i], 'label': labels[i]});
    }

    result.sort((b, a) => a['prob'].compareTo(b['prob']));
    

    return result;
  }
}
