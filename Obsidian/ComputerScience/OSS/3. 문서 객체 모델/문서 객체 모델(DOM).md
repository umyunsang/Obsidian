
---
## DOM 이해하기
#### HTML 태그가 객체 형태인 이유
![[Pasted image 20241108104046.png]]
#### 웹 시스템 관련 객체 (가장 상위 객체는 window)
- **Array, String, Math, Date 객체 **
	- 데이터를 보관하고 처리하도록 자바스크립트에서 제공하는 기본 객체 (수업자료 2장 참고) 
- **문서 객체모델(DOM)의 객체** 
	- 자바스크립트가 손쉽게 객체에 접근하여 읽고 조작하도록 제공됨
	- DOM의 가장 상위는 document 객체 
- **event 객체** 
	- 이벤트 발생시 생성되며 이벤트 관련 많은 정보를 담고 있음 (수업자료 1장 참고) 
- **기타 브라우저 객체모델(BOM)의 객체** 
	- 웹브라우저와 관련된 내용을 객체 형태로 만든 것 (soon)

## DOM 객체 다루기 : HTML 요소 접근하기
![[Pasted image 20241108104431.png]]

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
>![[Pasted image 20241108110347.png]]
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
>![[Pasted image 20241108110531.png]]
- st : textarea를 가르키는게 아니고 textarea 안에 내용물을 가져온다
---

