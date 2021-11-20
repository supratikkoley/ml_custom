#import "MlCustomPlugin.h"
#if __has_include(<ml_custom/ml_custom-Swift.h>)
#import <ml_custom/ml_custom-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "ml_custom-Swift.h"
#endif

@implementation MlCustomPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftMlCustomPlugin registerWithRegistrar:registrar];
}
@end
