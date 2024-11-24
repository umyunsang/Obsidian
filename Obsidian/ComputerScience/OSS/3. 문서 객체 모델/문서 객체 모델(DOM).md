
---
## DOM 이해하기
#### HTML 태그가 객체 형태인 이유
![](../../../../image/Pasted%20image%2020241108104046.png)
#### 웹 시스템 관련 객체 (가장 상위 객체는 window)
- **Array, String, Math, Date 객체**
	- 데이터를 보관하고 처리하도록 자바스크립트에서 제공하는 기본 객체 (수업자료 2장 참고) 
- **문서 객체모델(DOM)의 객체** 
	- 자바스크립트가 손쉽게 객체에 접근하여 읽고 조작하도록 제공됨
	- DOM의 가장 상위는 document 객체 
- **event 객체** 
	- 이벤트 발생시 생성되며 이벤트 관련 많은 정보를 담고 있음 (수업자료 1장 참고) 
- **기타 브라우저 객체모델(BOM)의 객체** 
	- 웹브라우저와 관련된 내용을 객체 형태로 만든 것 (soon)

## DOM 객체 다루기 : HTML 요소 접근하기
![](../../../../image/Pasted%20image%2020241108104431.png)

---
#### 예제1. HTML 요소 1개에 접근하기 (화면 결과는?)
```html
<body>
    <p id="ice">딸기</p>
    <p id="ice">쵸코</p>
    
    <script>
        document.getElementById("ice").innerHTML = "아이스크림";
    </script>
</body>
```
- 딸기 -> 아이스크림으로 변경 됨, 초코는 그대로 출력 됨
---
#### 예제2. HTML 요소 여러 개에 접근하기 (화면 결과는?)
```html
<body>
    <p>carrot 아니다</p>
    <p class="carrot">carrot 1</p>
    <p class="carrot">carrot 2</p>
    <hr>
    
    <script>
        const carrotarr = document.getElementsByClassName("carrot");
        const tagarr = document.getElementsByTagName("p");
        
        for (let i = 0; i < carrotarr.length; i++) {
            document.write(`carrotarr[${i}] = ${carrotarr[i].innerHTML}<br>`);
        }
        
        document.write("<hr>");
        for (let i = 0; i < tagarr.length; i++) {
            document.write(`tagarr[${i}] = ${tagarr[i].innerHTML}<br>`);
        }
    </script>
</body>
```
>[!출력결과]
>carrot 아니다
>carrot 1
>carrot 2
>carrotarr[0] = carrot 1
>carrotarr[1] = carrot 2
>tagarr[0] = carrot
>tagarr[1] = carrot 1
>tagarr[2] = carrot 2
- getElementsByClassName, getElementsByTagName 꼭 기억
---
#### 예제3. HTML 요소 여러 개에 태그선택자로 접근하기 (화면 결과는?)
```html
<body>
    <p class="carrot">carrot 1</p>
    <p class="carrot">carrot 2</p>
    <p>carrot도 ice도 아니다</p>
    <p id="ice">ice</p>
    <hr>
    
    <script>
        const carrotarr = document.querySelectorAll(".carrot");
        for (let i = 0; i < carrotarr.length; i++) {
            document.write(`carrotarr[${i}] = ${carrotarr[i].innerHTML}<br>`);
        }
        
        document.write("<hr>");
        
        const icearr = document.querySelectorAll("#ice");
        for (let i = 0; i < icearr.length; i++) {
            document.write(`icearr[${i}] = ${icearr[i].innerHTML}<br>`);
        }
    </script>
</body>
```
>[!출력결과]
>carrot 1
carrot 2
carrot도 ice도 아니다
ice
carrotarr[0] = carrot 1  
carrotarr[1] = carrot 2  
icearr[0] = ice
- querySelectorAll(".carrot") : 복수로 받을때 .(점)이 필요
- querySelectorAll("#ice") : 하나만 받을때 #(샵)이 필요
---
#### \<form> 내부 경우 기존과는 다르게 접근 가능
```html
<form name="myform">
	<input type="text" name="userId">
</form>
<script>
	const id1 = document.myform.userId;
</script>
```
---
#### 예제4. form 안의 input 문자열 읽기
```html
<body>
    <form name="myform">
        사용자ID <input type="text" name="userId"><br>
        패스워드 <input type="password" name="password"><br>
        <input type="submit" value="log in" onClick="getInput(); return false;">
    </form>
    
    <script>
        function getInput() {
            const id = document.myform.userId;
            const pw = document.myform.password;
            
            if (pw.value.length <= 4) {
                alert(`${id.value}님 패스워드를 변경하세요`);
            } else {
                alert(`${id.value}님 환영합니다`);
            }
        }
    </script>
</body>
```
>[!출력결과]
>![](../../../../image/Pasted%20image%2020241108110347.png)
- const id = document.myform.userId , const pw = document.myform.password 가능하다
---
#### 예제5. textarea 문자열 읽기
```html
<body>
    <textarea id="text" rows="5" cols="20"></textarea><br>
    <input type="button" value="Get Text" onClick="getTextarea()">
    <hr>
    <p id="result"></p>
    
    <script>
        function getTextarea() {
            const st = document.getElementById("text").value;
            document.getElementById("result").innerHTML = st;
        }
    </script>
</body>
```
>[!출력결과]
>![](../../../../image/Pasted%20image%2020241108110531.png)
- st : textarea를 가르키는게 아니고 textarea 안에 내용물을 가져온다
- textarea의 문자열은 getElementById로 받는다
---
#### 예제6. 체크박스 값 1개 읽기
```html
<body>
    <input type="checkbox" id="myCheckbox">본인 맞습니다.<br>
    <input type="button" value="다음" onClick="getCheckbox()">
    <hr>
    <p id="result"></p>
    
    <script>
        function getCheckbox() {
            if (document.getElementById("myCheckbox").checked == true) {
                document.getElementById("result").innerHTML = "작업을 진행하겠습니다.";
            } else {
                document.getElementById("result").innerHTML = "본인 인증 해주세요.";
            }
        }
    </script>
</body>
```
>[!출력결과]
>![](../../../../image/Pasted%20image%2020241108110924.png)
- 체크박스 값 1개를 읽을 때는 getElemnetById로 받는다
---
#### 예제7. 체크박스 값 여러 개 읽기
```html
<body>
    <input type="checkbox" name="myAdd" value="3000">추가샐러드 3000원<br>
    <input type="checkbox" name="myAdd" value="2000">추가음료수 2000원<br>
    <input type="checkbox" name="myAdd" value="500">추가피클 500원<br>
    <input type="button" value="계산" onClick="getCheckbox()">
    <hr>
    <p id="result"></p>
    
    <script>
        function getCheckbox() {
            let sum = 0;
            const addarr = document.getElementsByName("myAdd");
            for (let a of addarr) {
                if (a.checked) sum = sum + parseInt(a.value);
            }
            document.getElementById("result").innerHTML = `추가금액: ${sum}원`;
        }
    </script>
</body>
```
>[!출력결과]
>![](../../../../image/Pasted%20image%2020241108111310.png)
- name속성은 getElementsByName으로 받는다
---
#### 예제8. 리디오버튼에서 값 읽기
```html
<body>
    <input type="radio" name="color" value="red" checked>RED<br>
    <input type="radio" name="color" value="green">GREEN<br>
    <input type="radio" name="color" value="blue">BLUE<br>
    <input type="button" value="choice" onClick="getRadio()">
    <hr>
    <h2 id="result"></h2>
    
    <script>
        function getRadio() {
            const colorarr = document.querySelectorAll("[type='radio']");
            for (let c of colorarr) {
                if (c.checked) {
                    document.getElementById("result").innerHTML = `${c.value}`;
                    break;
                }
            }
        }
    </script>
</body>
```
>[!출력결과]
>![](../../../../image/Pasted%20image%2020241108111706.png)
- getElementsByName("color") 로 받을 수 도 있음
- querySelectorAll("[type='radio']")
	- 대괄호([])를 사용해서 type전부를 가져 올 수 있음
---
#### 예제9. 드롭박스에서 값 읽기
```html
<body>
    <p>키우고 싶은 동물은</p>
    <select id="selectBox">
        <option>==선택==</option>
        <option>dog</option>
        <option>cat</option>
        <option>duck</option>
        <option>koala</option>
    </select>
    <h2 id="result"></h2>
    
    <script>
        const s = document.getElementById("selectBox");
        s.addEventListener("change", getSelect);
        s.options[0].hidden = true;
        
        function getSelect() {
            const i = s.selectedIndex;
            document.getElementById("result").innerHTML = s.options[i].value;
        }
    </script>
</body>
```
>[!출력결과]
>![](../../../../image/Pasted%20image%2020241108111909.png)
- s.addEventListener("change", getSelect);
	- on change 아님, getSelect() 괄호 필요없다
- 옵션태그에 별도로 value값이 없으면 태그 옆의 값이 value가 된다
- const i = s.selectedIndex;
	- 인덱스 번호를 가져옴
---
#### 예제10. 문자열 회전시키기
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예10</title>
    <link rel="stylesheet" href="style.css">
    <style>
        #hone {
            color: blue;
        }
    </style>
</head>
<body style="text-align: center">
    <h1 id="hone">Hello,Bye!</h1>
    <script>
        const h = document.getElementById("hone");
        setInterval(rotateString, 500);

        function rotateString() {
            h.textContent = h.textContent.slice(1) + h.textContent[0];
        }
    </script>
</body>
</html>
```
>[!출력]
>![](../../../../image/20241124-0652-05.0386461.mp4)
- h 변수 : string이 아니라 \<h1> 객체 자체를 가르킴
- setInterval(rotateString, 500) : rotateString() 함수를 0.5초 마다 실행함
	- 콜백함수 (매개인자로 함수명 괄호없이 이름만 사용)

---
#### 예제11. 속성 변경 (사진 체인지)
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예11</title>
    <link rel="stylesheet" href="style.css">
</head>
<body style="text-align: center">
    <img id="image" src="./static/photo1.jpg">
    <br>
    <button onclick="changeImage()">사진 변경</button>
    <script>
        let toggle = 1;
        const i = document.getElementById("image");

        function changeImage() {
            if (++toggle % 2 == 0)
                i.src = "./static/photo2.jpg";
            else
                i.src = "./static/photo1.jpg";
        }
    </script>
</body>
</html>
```
>[!출력]
>![](../../../../image/20241124-0653-54.0598573.mp4)

---
#### 예제12. 스타일 변경 (색, 모서리)
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예12</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="box" style="width:200px; height:200px; margin:auto;"></div>
    <script>
        const box = document.getElementById("box");
        const letters = "0123456789ABCDEF";
        let count = 0;
        const intervalID = setInterval(changeFigure, 300);

        function changeFigure() {
            if (count > 10) clearInterval(intervalID);
            const a = letters[Math.floor(Math.random() * 16)];
            const b = letters[Math.floor(Math.random() * 16)];
            box.style.background = '#00' + a + b + '00';
            box.style.borderRadius = `${count * 5}%`;
            count++;
        }
    </script>
</body>
</html>
```
>[!출력]
>![](../../../../image/20241124-0649-55.3171642.mp4)

---
#### 예제13. 노드 추가하기
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예13</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h3 id="h">사이트 추가하기</h3>
    <input type="button" value="add" onClick="addNode()">
    <script>
        const urlarr = ['https://www.daum.net', 'https://www.naver.com', 'https://www.donga.ac.kr'];
        const namearr = ['다음', '네이버', '동아대학교'];
        let i = 0;

        function addNode() {
            const node = document.createElement("a");
            node.href = urlarr[i];
            node.innerHTML = `<hr>${namearr[i]}`;
            document.getElementById("h").appendChild(node);
            if (++i >= urlarr.length) i = 0;
        }
    </script>
</body>
</html>
```

---
#### 예제14. 노드 삭제하기
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예14</title>
    <link rel="stylesheet" href="style.css">
    <style>
        #b4 {
            background: pink;
        }
    </style>
</head>
<body>
    <div id="d">
        <button id="b1">첫 번째</button>
        <button id="b2">두 번째</button>
        <button id="b3">세 번째</button>
    </div>
    <button id="b4" onClick="removeNode()">노드 지우기</button>
    <script>
        let num = 1;

        function removeNode() {
            const parent = document.getElementById("d");
            const child = document.getElementById(`b${num}`);
            if (child) {
                parent.removeChild(child);
                num++;
            }
        }
    </script>
</body>
</html>

```


---
#### 예제15. 행 추가 및 삭제
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예15</title>
    <link rel="stylesheet" href="style.css">
    <style>
        table, td {
            border-collapse: collapse;
        }
    </style>
</head>
<body>
    <table id="tbl">
        <tr>
            <td>원래 셀</td>
            <td>원래 셀</td>
        </tr>
    </table>
    <br>
    <button onClick="changeTable(1)">행 추가</button>
    <button onClick="changeTable(-1)">행 삭제</button>
    <script>
        const t = document.getElementById("tbl");

        function changeTable(gubun) {
            const num = t.rows.length;
            if (gubun == 1) {
                const newRow = t.insertRow(num);
                const cell1 = newRow.insertCell(0);
                const cell2 = newRow.insertCell(1);
                cell1.innerHTML = `추가 셀 ${num}`;
                cell2.innerHTML = `추가 셀 ${num}`;
            } else {
                if (num > 1) {
                    t.deleteRow(num - 1);
                } else {
                    alert("맨 위 행은 삭제 안 해 드림!!");
                }
            }
        }
    </script>
</body>
</html>

```

---
