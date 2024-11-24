
---
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