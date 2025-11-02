---
title: Flutter에서 Nested ListView보단 Sliver와 ScrollView를 사용해야하는 이유
date: 2022-09-17 09:58:28.000000000 -04:00
categories: flutter
redirect_to: https://www.jaebok-lee.com/posts/ko/flutter-nested-listview
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "left"
});
</script>


## Nested ListView

Flutter에서 [ListView](https://api.flutter.dev/flutter/widgets/ListView-class.html)는 굉장히 많이 사용되는 위젯이다.
그런데 가끔은 그냥 Flat List가 필요한 것이 아니라 Nested ListView가 필요할 때가 생긴다.
가장 쉬운 예시를 들자면 댓글/대댓글이 있을 수 있겠다.

인스타그램을 살펴보자.

<img src="https://img.insight.co.kr/static/2019/09/18/700/i87r6b4q3r97064hy8a9.jpg" width=500>

위 그림과 같이, 댓글(Outer ListView) 안에 대댓글(Inner ListView)이 구성되어, ListView를 중첩해서 사용해야한다.

## ListView in ListView

쉽게 생각하면 '뭐야 그냥 ListView 2개 그리면 되는거 아닌가?'라고 생각할 수 있다.
실제로 Nested ListView in Flutter 라고만 검색을 해보면 쉽게 이런 스택오버플로우의 [답변](https://stackoverflow.com/questions/45270900/how-to-implement-nested-listview-in-flutter)을 볼 수도 있다.
그냥 Inner ListView에 `shrinkWrap=true`와 `ClampingScrollPhysics()`라는 파라미터를 사용하면 된다는 것이다.

그렇다면 실제로 이러한 방식으로 댓글/대댓글을 구현하면 어떤 문제가 생길 수 있을까?

<img src="https://imgur.com/dfyQpOo.jpg" width=500>

위 사진은 Flutter를 이용하여 간단한 UI를 그린 모습이다.
코드도 정말 심플한데,

```dart
@override
Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
            appBar: AppBar(title: const Text("Flutter ListView Demo")),
            body: ListView.builder(itemBuilder: (context, commentIndex) {
                return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                        Text("Comment: $commentIndex"),
                        ListView.builder(itemBuilder: (context, replyIndex) {
                            debugPrint("Reply $replyIndex in Comment $commentIndex is generated!!");
                            return Text("Reply: $replyIndex");
                        },
                        itemCount: 100,
                        shrinkWrap: true,
                        physics: const ClampingScrollPhysics(),),
                    ],
                );
            },
            itemCount: 10,
            ),
        ),
    );
}
```

그냥 시키는대로 하면 위 UI를 쉽게 구현할 수 있다.
하지만 큰 문제점이 하나 있다.
콘솔창에서 `debugPrint`에 해당하는 부분이 어떻게 출력되는지 확인하면 알 수 있는데,

<img src="https://imgur.com/WH7IeED.jpg" width=500>

화면에는 Reply(대댓글)이 36번까지밖에 보이지 않지만, 실제로는 Inner ListView에 있는 Reply가 모두 (100개) 한 번에 생성된다는 점이다.
실제로 `ListView.builder()`를 이용하여 List를 생성하는 것은 화면에 '스크롤될 때마다' 아이템이 생성되는 방식으로 실제 보여질 때 Lazy하게 UI를 그릴 수 있다는 장점이 있는 것인데, 이것을 Nested ListView 구조는 불가능하게 만들어버린다.
이는 Outer ListView에서 UI 공간 확보를 위해 Inner ListView의 크기를 미리 파악하기 위해 사전에 build를 모두 해버려서 발생하는 일이다.

## Sliver & CustomScrollView

이러한 문제를 해결하는 것은 바로 `CustomScrollView`와 `SliverList`를 사용하는 것이다.
[Sliver](https://medium.com/flutter/slivers-demystified-6ff68ab0296f)는 쉽게 말하면, 스크롤이 가능한 부분의 일부를 의미한다.
실제 `ListView`와 `GridView`도 `Sliver`를 이용하여 구현되어있다.

따라서 `Sliver`를 직접 사용하는 경우는 복잡한 List를 여러 개 엮어서 Scroll 가능하게 만드는 경우이다.
아래로 내릴 때 AppBar는 남아있게 또는 축소되게하거나, `Grid`와 `List`를 마구 섞어쓸 때 Customize하기에 매우 훌륭한 도구이다.
그리고 이렇게 편리한 `Sliver`는 `CustomScrollView` 안에서 사용할 수 있다.

어쨋든 이 글은 `Sliver` 자체를 설명하려는 것은 아니기 때문에 이들을 이용한 해결 코드를 보면 아래와 같다.

```dart
@override
Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
            appBar: AppBar(title: const Text("Flutter ListView Demo")),
            body: CustomScrollView(
                slivers: [
                    for (int commentIndex = 0; commentIndex < 10; commentIndex++) ...[
                        SliverToBoxAdapter(child: Text("Comment: $commentIndex")),
                        SliverList(
                        delegate: SliverChildBuilderDelegate(((context, replyIndex) {
                            debugPrint("Reply $replyIndex in Comment $commentIndex is generated!!");
                            return Text("Reply: $replyIndex");
                        }),
                        childCount: 100,),
                    ),
                    ],
                ],
            )
        ),
    );
}
```

작동 원리는 매우 간단한데, Comment 갯수만큼 for문을 돌면서 Reply를 만드는 `SliverList`를 넣어주는 것이다.
쉽게 말하자면, 기존의 Nested ListView구조를 해체해서 Flat한 SliverList의 List로 만들어주는 작업으로도 볼 수 있다.

for문 뒤에 spread operator `...`를 사용하여 [[sliver, sliver], [sliver, sliver]] 를 [sliver, sliver, sliver, ...]과 같이 펴는 코드를 확인할 수 있다.
실제로 `debugPrint`부분을 출력해보면,

<img src="https://imgur.com/VdCUxCY.jpg" width=500>

100번째 Reply가 아닌 52번째 Reply까지만 생성되었고, 스크롤에 따라서 Lazy하게 아이템들이 생성되는 것을 확인할 수 있었다.
얼핏 코드만 보기에는 한 번에 모든 댓글/대댓글을 생성하는 것처럼 보이지만 `SliverChildBuilderDelegate`를 사용한 Reply들은 UI에 보여질 때 생성되게 된다.

## References
- https://stackoverflow.com/questions/45270900/how-to-implement-nested-listview-in-flutter
- https://timm.preetz.name/articles/nested-listviews-in-flutter
- https://github.com/flutter/flutter/issues/26072
- https://medium.com/flutter/slivers-demystified-6ff68ab0296f