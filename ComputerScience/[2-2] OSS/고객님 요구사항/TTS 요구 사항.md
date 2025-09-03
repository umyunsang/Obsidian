
---
## TTS원본.html
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TTS원본</title>
</head>
<body>
    <h2>텍스트를 읽어드립니다</h2>
    <textarea id="box" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <button onclick="readText()">읽기</button>
    
    <script>
        function readText() {
            const a = document.getElementById("box");
            let text = a.value.trim();
            if (text === "") {
                alert("읽을 텍스트를 입력하세요.");
                return;
            }
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    </script>
</body>
</html>
```

#### 고객님의 요구사항
- 입력박스를 2개 만들고 디자인은 아래 화면처럼 해주세요. (가운데 정렬, 맨 위 파란 글자, 박스 내 글자크기, 버튼 변경(색, 글자크기, 여백) 
- 한 박스에라도 글자 입력 시에는 읽어주세요. 두 박스 모두 비면 팝업창 띄워주세요. 
- 위 박스부터 차례대로 텍스트 읽어주세요.
![](../../../../image/Pasted%20image%2020241124151009.png)

#### 팀장의 요구사항 
- (1) 박스가 2개 뿐이므로 반복문 사용하지 말고 getElementById() 2번 사용해서 편하게 코딩한 소스 TTS수정후.html 
- (2) 지금은 2개지만 앞으로 확장될 가능성을 대비해서 getElementsByClassName() 사용해서 배열로 받은 후 for-of 반복문으로 코딩한 소스 (TTS수정후for.html)

>[!TTS수정후.html]
```html
<style>
    body {text-align: center;}
    h2 {color: blue;}
    button {background-color: lime;
            font-size: 1em;
            padding: 0px 20px 0px 20px;}
</style>
<body>
    <h2>여러 텍스트를 이어서 읽어드립니다</h2>
    <textarea id="box1" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <textarea id="box2" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <button onclick="readText()">읽기</button>
<script>
function readText() {
    const a = document.getElementById("box1");
    const b = document.getElementById("box2");
    let text_a = a.value.trim();
    let text_b = b.value.trim();
    if (text_a == "" && text_b == "") {
        alert("읽을 텍스트를 입력하세요.");
        return;
    }
    const utterance_a = new SpeechSynthesisUtterance(text_a);
    const utterance_b = new SpeechSynthesisUtterance(text_b);
    window.speechSynthesis.speak(utterance_a);
    window.speechSynthesis.speak(utterance_b);
}
</script>
</body>
```

>[!TTS수정후for.html]
```html
<body>
    <h2>여러 텍스트를 이어서 읽어드립니다</h2>
    <textarea class="box" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <textarea class="box" rows="5" cols="50" placeholder="여기에 입력하세요"></textarea><br>
    <button onclick="readText()">읽기</button>
    
<script>
    function readText() {
    const arr = document.getElementsByClassName("box");
    let main_text = "";

    for (let i = 0; i < arr.length; i++) {
        let text = arr[i].value.trim();
        main_text += text;
    }
    if (main_text !== "") {
        const utterance = new SpeechSynthesisUtterance(main_text);
        window.speechSynthesis.speak(utterance);
    }
    else {
        alert("읽을 텍스트를 입력하세요.");
        return;
    }
    }
</script>
</body>
```