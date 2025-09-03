
---
## 1.html
	1.&nbsp;대신 <style> 태그에서 공백을 넣을 수 있음
```html
<!doctype html>
<!-- 나는 주석입니다 -->
<html>
	<head>
		<meta charset="UTF-8">
		<title>내가 CEO다~</title>
	</head>
	<style>
		a { margin-right : 40px;}
	<body>
	<h2>James 주식회사</h2><hr><br>
	<!-- 회사명은 마음대로 정하세요 -->
	<a href="2.html">로그인</a>
	&nbsp;&nbsp;&nbsp;&nbsp;
	<a href="3.html">회원가입</a>
	&nbsp;&nbsp;&nbsp;&nbsp;
	<a href="https://www.naver.com" target="_blank">네이버</a>
	</body>
</html>
```

## 2.html
	1.<body style="background-color:silver"> style 속성으로도 사용 가능
	2.<strong>태그 글자를 진하게 
	3.input type= "text": 문자열 입력, "password": 화면에 기호로 표현
	4.name속성은 나중에 서버로 데이터를 보낼 때 매우 중요
	
```html
<html><head> <title>로그인</title></head> 
<style>
  body {
   background-color:silver;
  }
</style>
<body>
	<form action="" method="" >
		<strong>아이디와 비밀번호를 입력하세요</strong>
	    <p>
	    아이디: <input type="text" name="id"> <br>
	    비밀번호: <input type="password" name="pw">
	   </p>
	   <button type="submit">확인</button>
	</form>
</body>
</html>

```

## 3.html
	1.<select> 태그는 여러 데이터 중 하나를 선택
	2.type 속성의 값이 reset인 버튼은 화면 초기화 기능
	3.value 속성도 나중에 서버로 데이터를 보낼 때 중요
```html
<html><head><title>회원가입 화면이다~</title></head>
<style>
	button { margin-right : 25px; }
</style>
<body>
	<h3> James 주식 회사 회원 가입 </h3>
	<form action="" method="" >
		<p>
	    아이디: <input type="text" name="id"> <br>
	    비밀번호: <input type="password" name="pw"> <br>
	    이름: <input type="text" name="person"> <br>
	    성별: <input type="radio" name="gender" value="man">맨
	    <input type="radio" name="gender" value="woman">워먼<br>
	    직업: <select name="job">
	    <option>아직고딩</option> <option>21세기대학생</option>
	    <option>아티스트</option> <option>공유</option>
	    </select>
		</p>
		<button type="submit">등록</button> <button type="reset">취소</button>
	</form>
</body>
</html>
```

## 4.html
	1.<table> 태그
		<tr>로 시작 한줄 안에 한칸은 <th>,<td>
		<th>은 <td>보다 진하고 가운데 정렬
		border 속성은 테두리를 결정
	2.<span> 태그
		별도의 태그가 없는 구간을 정할 때 사용
		<style> 태그에서 span 사용 가능 함
```html
<html><head><title>리스트</title></head>
<style>
	body { background-color: yellow; }
	span { color:gray; font-size:0.8em; }
	table, th, td { border: 1px solid black;}
	</style>
<body>
	<h2>회원리스트(관리자용)</h2>
	<table>
	    <tr> <th>아이디 <th>비밀번호 <th>이름 <th>성별 <th>직업
	    <tr> <td>sony <td>1111 <td>손흥민 <td>남 <td>축구선수
	    <tr> <td>sunflower <td>2222 <td>고흐 <td>남 <td>화가
	    <tr> <td>&nbsp; <td>&nbsp; <td>&nbsp; <td>&nbsp; <td>&nbsp;
	</table>
	<span>총 회원 수: 2명</span>
</body>
</html>
```