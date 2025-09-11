---
title: Flutter에서 Android 액티비티를 실행하는 방법
date: 2022-09-17 09:58:28.000000000 -04:00
categories: flutter
redirect_to: https://jaebok-lee.com/posts/flutter-android
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

---

`lib/main.dart`

{% highlight dart linenos %}
import 'package:flutter/material.dart';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter - Android Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'Flutter - Android Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  
  // MethodChannel
  static const platform = MethodChannel("com.example.example/message");

  // Message
  String _message = "Initial message";

  // Invoke Method
  Future<void> _getMessage() async {
    String message;
    try {
      message = await platform.invokeMethod('getMessageAndroid');
    } on PlatformException {
      message = "Failed to get message from Android";
    }

    setState(() {
      _message = message;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(_message),
            ElevatedButton(onPressed: _getMessage, child: const Text("Get message"))
          ],
        ),
      ),
    );
  }
}
{% endhighlight %}

먼저, 플러터에서는 기본 샘플 앱을 수정하여 안드로이드 플랫폼을 호출하기 위한 버튼과 반환되는 메세지를 표시하는 간단한 UI를 만들었다.

- `line 37`: 플러터와 안드로이드간의 플랫폼 채널로 `MethodChannel`을 선언한다. 채널명은 하나의 앱 안에서 유니크해야하기 때문에 prefix로 domain을 사용하고 뒤에 특정 기능을 명시하는 방법이 좋아보인다.
- `line 43`: 버튼을 눌렀을 때 실행되는 함수이다. 플랫폼채널의 `invokeMethod`를 통해 플랫폼의 native code를 수행하며 어떤 동작을 실행해야할지 명시하기위해 String 값을 인자로 넘겨준다.

---

`MainActivity.java`

{% highlight java linenos %}
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
{% endhighlight %}

플러터에서 `MethodChannel`을 통해 플랫폼 API를 호출했을 경우, 이 API 호출을 담당하는 `MainActivity`의 코드이다.
이 `MainActivity`는 UI layout이 존재하지 않으며, 실제 UI는 `SecondActivity`에서 관리하고 여기서는 해당 Activity를 호출하는 역할을 담당한다.

- `line 11`: 기본적으로 `MainActivity`는 플러터 엔진과의 연결을 위해 `FlutterActivity`를 상속받는다.
- `line 13`: `lib/main.dart`에서 정의한 플랫폼 채널의 이름을 동일하게 사용하여야 한다.
- `line 16`: `MethodChannel`을 이용한 통신의 결과를 전달하기 위한 `MethodChannel.Result` 변수이다.
- `line 33`: `MethodChannel`을 이용하여 통신하는 부분이다. `line 39`을 보면 알 수 있듯, 플러터에서 플랫폼 채널을 이용해 호출한 String 값을 확인한 뒤에 어떤 행동을 할지 결정하게 된다.
- `line 22`: UI가 있는 `SecondActivity`를 호출하기 위한 코드이다.
- `line 54`: `SecondActivity`가 호출되고 결과(이 예제에서는 `EditText`의 값)를 반환할텐데, 이 반환이 일어났을 경우 실행되는 코드이다. `REQUEST_CODE`는 결과가 반환되었을 때 어떤 Activity에서 이 결과가 반환되었는지 구분하기 위해 사용되는 코드이다. 여기서는 `SecondActivity`로부터 String 결과를 받아 `myResult`변수를 통해 Flutter로 반환하게 된다.

---

`SecondActivity.java`

{% highlight java linenos %}
package com.example.example;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import androidx.appcompat.app.AppCompatActivity;

public class SecondActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);

        EditText editText = findViewById(R.id.editText);

        // Return android-side message
        Button button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent data = new Intent();
                data.setData(Uri.parse(editText.getText().toString()));
                setResult(RESULT_OK, data);
                finish();
            }
        });
    }
}
{% endhighlight %}

아주 간단하게 `EditText`와 `Button` UI가 있고, `Button`이 클릭되면 `EditText`에 있는 값을 읽은 뒤 `line 28`을 통해 결과를 반환한다.

---

`activity_second.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:orientation="vertical"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:gravity="center_vertical">

    <EditText
        android:id="@+id/editText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:text="Type Message Here"/>

    <Button
        android:id="@+id/button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:text="Send Message"/>

</LinearLayout>
```

---

`android/app/build.gradle`

```gradle
dependencies {
    implementation 'androidx.appcompat:appcompat:1.3.0'
}
```

UI가 있는 새로운 `SecondActivity`를 추가하기 위해 이 dependency를 추가해준다.

---

`AndroidManifest.xml`

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.example">
   <application
        android:label="example"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop"
            android:theme="@style/LaunchTheme"
            android:configChanges="orientation|keyboardHidden|keyboard|screenSize|smallestScreenSize|locale|layoutDirection|fontScale|screenLayout|density|uiMode"
            android:hardwareAccelerated="true"
            android:windowSoftInputMode="adjustResize">
            <!-- Specifies an Android theme to apply to this Activity as soon as
                 the Android process has started. This theme is visible to the user
                 while the Flutter UI initializes. After that, this theme continues
                 to determine the Window background behind the Flutter UI. -->
            <meta-data
              android:name="io.flutter.embedding.android.NormalTheme"
              android:resource="@style/NormalTheme"
              />
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        
        <!-- HERE -->
        <activity android:name=".SecondActivity"
            android:parentActivityName=".MainActivity"
            android:theme="@style/Theme.AppCompat.Light">

            <meta-data
                android:name="android.support.PARENT_ACTIVITY"
                android:value=".MainActivity"/>
        </activity>
        <!-- HERE -->

        <meta-data
            android:name="flutterEmbedding"
            android:value="2" />
    </application>
</manifest>
```

새로운 Activity가 추가되었으니 `AndroidManifest.xml`에 추가해준다. \<\!-- HERE -->로 표시된 부분에서 확인할 수 있다.
    
## 결론 ##

이렇게 Flutter 내에서 Android native code 실행을 위한 플랫폼 채널의 사용법을 확인해보았다.
예제 코드를 전체 확인하고 싶으면 [여기](https://github.com/zzaebok/flutter-android-example)에서 확인해보길 바란다.

## Reference ##

- https://flutter-ko.dev/docs/development/platform-integration/platform-channels
- https://developer.android.com/training/basics/intents/result
