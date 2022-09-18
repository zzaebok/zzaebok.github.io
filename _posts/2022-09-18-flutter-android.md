---
title: "Flutter에서 Android 액티비티를 실행하는 방법"
date: 2022-09-17 09:58:28 -0400
categories: flutter
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>


## Android native code 실행하기 ##

Flutter는 안드로이드, IOS, 웹 등 cross platform에서 하나의 코드로 앱을 개발하기 쉽도록 도와주는 도구이다.
하지만, 플러터에서 사용할 수 있는 플러그인들이 모든 플랫폼에서 가능한 모든 기능들을 담고 있을 수 없다.
따라서 플러터 내에서도 특정 플랫폼에서 제공하는 기능들을 사용하기 위해 native code를 직접 작성해야할 일이 생기기 마련이다.
오늘은 플러터에서 Java로 작성된 Android native code를 실행하여 Activity를 실행하고 그 결과를 반환하는 예제를 만들어보고자 한다.

<img src="https://docs.flutter.dev/assets/images/docs/PlatformChannels.png" width="600">

플러터는 특정 플랫폼의 API를 이용하기 위하여 메세지 패싱 방식을 이용한다.
플랫폼 채널을 이용하여 host (안드로이드, IOS)에 메세지를 보내 특정 플랫폼의 기능을 호출하고 응답을 받는 것이다.
위 사진은 플랫폼 채널의 아키텍처 오버뷰이다.
메세지가 플랫폼 채널을 이용하여 플러터 앱(UI)과 호스트(플랫폼) 사이에서 전달되는 것을 확인할 수 있다.

이번 예제에서는 플랫폼 중 자바로 작성된 안드로이드 플랫폼과의 통신 코드를 작성할 것이다.
단, 예제인만큼 특별한 기능을 이용하지는 않고 안드로이드 액티비티를 호출하고 해당 안드로이드 액티비티에서 `EditText`내용을 반환하는 내용을 준비하였다.

<img src="https://imgur.com/DG4oJ86.png" width="170"> <img src="https://imgur.com/99OLcc6.png" width="170"> <img src="https://imgur.com/A8IXrv1.png" width="170"> <img src="https://imgur.com/4r6iQkc.png" width="170">

위 예제에서 첫 번째와 마지막 스크린샷은 Flutter앱, 두 번째와 세 번째 스크린샷은 Android Activity의 모습이다.
순서는

1. Flutter 앱에서 `Get Message`버튼을 클릭해 Android Activity 호출하기
2. Android Activity의 `EditText` 부분을 수정하기
3. Android Activity의 `Send Message`버튼을 클릭해 수정된 `EditText`내용을 Flutter앱에 전달하기
4. Android Activity로부터 받아온 메세지를 Flutter에서 보여주기 (나는 문어)

## 코드 살펴보기 ##

위 예제는 일단 flutter 앱 예제를 생성하는 것으로 시작한다.
Android native code로는 Java를 사용할 것이므로 아래와 같이 명령어를 입력하여 앱을 생성한다.

```bash
flutter create example -a java
```

변경해야할 파일은 총 6개로 이다.

- `lib/main.dart`
- `android/app/src/main/java/com/example/example/MainActivity.java`
- `android/app/src/main/java/com/example/example/SecondActivity.java`
- `android/app/src/main/res/layout/activity_second.xml`
- `android/app/build.gradle`
- `android/app/src/main/AndroidManifest.xml`

실제로 플랫폼 채널은 플러터 `lib/main.dart`와 안드로이드 `MainActivity.java` 사이에 형성이 되고, 위 예제의 화면이 되는 `SecondActivity`가 `MainActivity`에서 호출되는 형태이다.
코드를 살펴보자

```java
package com.example.example;

import android.content.Intent;

import androidx.annotation.NonNull;
import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodChannel;

public class MainActivity extends FlutterActivity {
    // Channel name
    private static final String CHANNEL = "com.example.example/message";

    // Result variable
    private MethodChannel.Result myResult;

    // Request code
    private static final int REQUEST_CODE = 1234;

    // Invoked method
    private void getMessageAndroid() {
        Intent intent = new Intent(this, SecondActivity.class);
        startActivityForResult(intent, REQUEST_CODE);
    }
    
    // Configure flutter engine
    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        super.configureFlutterEngine(flutterEngine);

        // Method channel
        new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), CHANNEL)
            .setMethodCallHandler(
                (call, result) -> {
                    myResult = result;

                    // Invoked method handling
                    if (call.method.equals("getMessageAndroid")) {
                        try {
                            getMessageAndroid();
                        } catch (Exception e) {
                            myResult.error("Unavailable", "Opennig SecondActivity is not available", null);
                        }
                    } else {
                        myResult.notImplemented();
                    }
                }
            );
    }

    // Activity result from invoked method
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                String resultString = data.getData().toString();
                myResult.success(resultString);
            } else {
                myResult.error("Unavailable", "Result from SecondActivity is not OK", null);
            }
        }
    }
}
```

## Reference ##

- https://flutter-ko.dev/docs/development/platform-integration/platform-channels
- https://developer.android.com/training/basics/intents/result
