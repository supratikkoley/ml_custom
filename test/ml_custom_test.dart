import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
// import 'package:ml_custom/ml_custom.dart';

void main() {
  const MethodChannel channel = MethodChannel('ml_custom');

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    channel.setMockMethodCallHandler((MethodCall methodCall) async {
      return '42';
    });
  });

  tearDown(() {
    channel.setMockMethodCallHandler(null);
  });

  // test('getPlatformVersion', () async {
  //   expect(await MlCustom., '42');
  // });
}
