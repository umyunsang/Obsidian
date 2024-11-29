
---
## slot원본.html
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>슬롯머신</title>
  <link rel="stylesheet" href="slot.css">
</head>
<body>
  <div class="slot-machine">
    <span class="slot" id="slot1">🍒</span>
    <span class="slot" id="slot2">🍋</span>
    <span class="slot" id="slot3">🍊</span>
  </div>
  <button onclick="startSlotMachine()">슬롯머신 돌리기</button>
  <h3 id="result">ready</h3>
  <script>
    const symbols = ["🍒", "🍋", "🍊", "⭐", "🔔"];
    const slotarr = document.querySelectorAll(".slot");
    const r = document.getElementById("result");

    function getRandomSymbol() {
      return symbols[Math.floor(Math.random() * symbols.length)];
    }

    function startSlotMachine() {
      const finalSymbols = [];
      for (let i = 0; i < slotarr.length; i++) 
        finalSymbols.push(getRandomSymbol());
      
      let iterationCount = 0;

      function spinSlots() {
        iterationCount++;
        slotarr.forEach((s) => {s.innerHTML = getRandomSymbol();});

        if (iterationCount >= 10) {
          clearInterval(intervalId);
          slotarr.forEach((s, i) => { s.innerHTML = finalSymbols[i];});

          /* 결과 출력 */
          if (finalSymbols[0] == finalSymbols[1] && finalSymbols[1] == finalSymbols[2]) 
            r.textContent = "Congratulation!!";
          else if (finalSymbols[0] == finalSymbols[1] || finalSymbols[1] == finalSymbols[2] || finalSymbols[0] == finalSymbols[2]) 
            r.textContent = "2개 맞았음";
          else 
            r.textContent = "꽝!";
        }
      }

      const intervalId = setInterval(spinSlots, 100); // 0.1초
    }
  </script>
</body>
</html>
```

#### 다른방법도 가능 (시험예상)
```js
const slotarr = document.querySelectorAll('.slot');
/* 1. getElementsByClassName() */
/* 2. getElementsTag__() */
```
#### 함수 A 안에 함수 a를 설정할 수 있습니다.
```js
function A(){
	...
	function a(){
		a()함수도 A()안에 있으므로
		A()의 지역변수도 공유할 수 있습니다.
		10번 반복되면 멈추라는 명령문도 여기 있음.
	}
	const intervalId = setInterval(a, 100);
}
```
- 버튼 onclick하면 A()함수가 돌아감 하지만 0.1초마다 반복되는 부분은 a()
