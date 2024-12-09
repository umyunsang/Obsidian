
---
#### Vanilla JS?  jQuery? React? Vue.js?
: 새롭게 시작하는 Front-end 개발 프로젝트에서 위 4가지 중 어느 쪽을 선택할 것인지 검토

- 현재 FE 개발자들의 가장 큰 관심사 중의 하나 
- 한 번 결정되면 되돌리기 어려움. 신중히 접근해야 
- 나무위키 등을 검색해도 내용이 잘 정리되어 있습니다.
---
#### Vanila JS, JQuery 단순 비교
```html
<body>
<h3>바닐라 JS 경우</h3>
	<ul id="list"> 
		<li>Item 1</li>
		<li>Item 2</li>
		<li>Item 3</li> 
	</ul> 
<script> 
	document.getElementById('list').style.background = 'lightgray';
	document.querySelectorAll('li').forEach(i => i.style.color = 'red');
	/* .forEach((i) => {i.style.color ='red'}); 와 동일*/
</script>
</body>
```
```html
<head>
<meta charset="UTF-8"><title>jQuery</title> 
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script></head>
<body>
<h3>jQuery를 사용했을 때</h3> 
	<ul id="list"> 
		<li>Item 1</li> 
		<li>Item 2</li>
		<li>Item 3</li>
	</ul> 
<script>
	$('#list').css('background', 'skyblue');
	$('li').css('color', 'red'); 
</script> 
</body>
```
