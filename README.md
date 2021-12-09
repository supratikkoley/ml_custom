# ml_custom

This plugin can only download ml model from firebase and can run the model on the image (it only works for image classification problem).

### Add dependency

```yaml
dependencies:
  ml_custom:
    git:
      url: git@github.com:supratikkoley/ml_custom.git
```

### Example

```dart
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:ml_custom/ml_custom.dart';

void main() {
  runApp(
    const MaterialApp(
      home: MyApp(),
    ),
  );
}

/// Widget with a future function that initiates actions from FirebaseML.
class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {

  List<Map<dynamic, dynamic>>? _classificationResult;

  final String _imgPath = ""; // image file path
  final String _labelPath = ""; // label file path

  String? _modelPath;


  /// Run model on an image and the consequent inference.
  Future<void> getImageLabels() async {
    try {
      debugPrint("Getting image labels...");

      await MlCustom.runModelOnImage(
        imgPath: _imgPath,
        modelPath: _modelPath,
        labelPath: _labelPath,
      ).then((result) {
        setState(() {
          _classificationResult = result.sublist(0, 10);
          print(_classificationResult);
        });
      });
    } catch (exception) {
      debugPrint("Failed on getting your image and it's labels: $exception");
      debugPrint('Continuing with the program...');
      rethrow;
    }
  }

  /// Gets the model ready for inference on images.
  Future<String> loadModel() async {
    final modelFile = await loadModelFromFirebase();
    return assignModelPath(modelFile);
  }

  static Future<File?> loadModelFromFirebase() async {
    try {
      // Create model with a name that is specified in the Firebase console
      const model = 'asana-detector';

      // Begin downloading and wait until the model is downloaded successfully.
      var modelFile = await MlCustom.getModel(model);

      assert(await MlCustom.isModelDownloaded(model) == true);

      // Get latest model file to use it for inference by the interpreter.
      modelFile = await MlCustom.getModel(model);
      assert(modelFile != null);
      return modelFile;
    } catch (exception) {
      debugPrint('Failed on loading your model from Firebase: $exception');
      debugPrint('The program will not be resumed');
      rethrow;
    }
  }

  Future<String> assignModelPath(File? modelFile) async {
    try {
      _modelPath = modelFile?.path;
      return 'Model is loaded';
    } catch (exception) {
      debugPrint(
          'Failed on loading your model to the TFLite interpreter: $exception');
      debugPrint('The program will not be resumed');
      rethrow;
    }
  }

  /// Shows image selection screen only when the model is ready to be used.
  Widget readyScreen() {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Firebase ML Custom example app'),
      ),
      body: Column(
        children: [
          // if (_image != null)
          Image.asset(_imgPath),
          // else
          //   const Text('Please select image to analyze.'),
          Column(
            children: _classificationResult != null
                ? _classificationResult!.map((res) {
                    return Text("${res["label"]}");
                  }).toList()
                : [],
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: getImageLabels,
        child: const Icon(Icons.search),
      ),
    );
  }

  /// In case of error shows unrecoverable error screen.
  Widget errorScreen() {
    return const Scaffold(
      body: Center(
        child: Text('Error loading model. Please check the logs.'),
      ),
    );
  }

  /// In case of long loading shows loading screen until model is ready or
  /// error is received.
  Widget loadingScreen() {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const <Widget>[
            Padding(
              padding: EdgeInsets.only(bottom: 20),
              child: CircularProgressIndicator(),
            ),
            Text('Please make sure that you are using internet.'),
          ],
        ),
      ),
    );
  }

  /// Shows different screens based on the state of the custom model.
  @override
  Widget build(BuildContext context) {
    return DefaultTextStyle(
      style: Theme.of(context).textTheme.headline2 ??
          const TextStyle(fontSize: 12),
      textAlign: TextAlign.center,
      child: FutureBuilder<String>(
        future: loadModel(), // a previously-obtained Future<String> or null
        builder: (BuildContext context, AsyncSnapshot<String> snapshot) {
          if (snapshot.hasData) {
            return readyScreen();
          } else if (snapshot.hasError) {
            return errorScreen();
          } else {
            return loadingScreen();
          }
        },
      ),
    );
  }
}
```
