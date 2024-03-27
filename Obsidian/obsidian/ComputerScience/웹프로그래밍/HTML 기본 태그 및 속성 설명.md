#ComputerScience #웹프로그래밍 #기본태그 

---
물론입니다! 주어진 HTML 코드의 각 태그를 자세히 설명하겠습니다.

1. `<!DOCTYPE html>`: HTML 문서가 HTML5를 사용한다는 것을 선언하는 문서 타입 선언자입니다.

2. `<html>`: HTML 문서의 루트 요소로서, 전체 HTML 내용을 감싸는 역할을 합니다.

3. `<head>`: HTML 문서의 메타데이터를 포함하는데 사용되며, 제목(title), 외부 스타일시트, 스크립트 등을 정의합니다.

4. `<title>`: 문서의 제목을 정의하는데 사용됩니다. 브라우저 탭에 표시됩니다.

5. `<body>`: 실제로 화면에 표시되는 내용을 포함하는데 사용되며, 텍스트, 이미지, 링크 등을 포함합니다.

6. `<h2>`: 두 번째 수준의 제목을 정의하는데 사용되며, 크기가 더 작은 제목입니다.

7. `<hr>`: 수평선을 삽입하는 태그입니다. 주로 문단이나 섹션의 구분선으로 사용됩니다.

8. `<p>`: 단락을 정의하는데 사용되며, 문단 단위의 텍스트를 포함합니다.

9. `<a>`: 하이퍼링크를 생성하는데 사용되며, href 속성에 지정된 URL로 이동합니다.

10. `<br>`: 줄바꿈을 생성하는 태그입니다. 인라인 요소로 사용되며, 닫는 태그가 없습니다.

11. `&nbsp;`: 비-breaking space를 의미하며, 공백을 삽입할 때 사용됩니다.

12. `<span>`: 특정 부분에 스타일을 적용하기 위해 사용되는 인라인 요소입니다.

13. `<strong>`: 텍스트를 강조하는데 사용되며, 일반적으로 굵은 글꼴로 표시됩니다.

14. `<input>`: 사용자 입력 필드를 생성하는데 사용되며, 텍스트 입력, 패스워드 입력 등 다양한 유형이 있습니다.

15. `<form>`: 사용자 입력 데이터를 서버로 제출하는데 사용되는 폼 요소를 정의합니다.

16. `<table>`: 표를 생성하는데 사용되며, 행(`<tr>`), 헤더 셀(`<th>`), 데이터 셀(`<td>`) 등을 포함합니다.

이렇게 각 태그는 문서의 구조를 정의하거나 특정한 기능을 수행하는 역할을 합니다.

```
<!DOCTYPE html>
<html>
<head>
    <title>HTML 실습</title>
</head>
<body style="background-color: silver;">
    <h2>제목</h2>
    <hr>
    <p>내용</p>
    <a href="https://www.naver.com" target="_blank">네이버</a>
    <br>
    <!--공백&nbsp;여러개&nbsp;삽입-->
    <span style="margin-right: 10px;">스타일 적용할 텍스트</span>
    <strong>진한 글자</strong>
    <input type="text" name="username" placeholder="유저명">
    <input type="password" name="password" placeholder="비밀번호">
    <form action="/submit" method="post">
        <input type="submit" value="전송">
        <input type="reset" value="초기화">
    </form>
    <table border="1">
        <tr>
            <th>헤더1</th>
            <th>헤더2</th>
        </tr>
        <tr>
            <td>데이터1</td>
            <td>데이터2</td>
        </tr>
    </table>
</body>
</html>
```

## 연습문제

```plaintext
Quiz #1: <!DOCTYPE html>의 의미는?
이것은 HTML 문서가 HTML5를 사용한다는 것을 선언하는 태그입니다.

Quiz #2: html태그 10가지 적어 보기
<head> <style> <body> <meta> <title> <a href> <table> <br> <hr> <form>

Quiz #3: html 주석 작성 방법은?
<!-- 나는 주석 입니다. -->

Quiz #4: ‘이동’이라는 글자를 누르면 새 탭이 열리면서 다른 페이지(new.html)로 링크하게 만드는 html 코딩 한 줄을 적어보세요.
<a href="new.html" target="_blank">이동

Quiz #5: type속성의 값이 submit인 button 태그는 화면에 버튼모양으로 보입니다. 이 버튼을 클릭하면 어떤 결과가 나올까요? 그리고 type속성의 값이 reset인 버튼을 클릭하면 어떤 결과가 나올까요?
submit : 해당 버튼이 포함된 <form> 요소의 데이터가 method 방법으로 action 주소에 제출됩니다.
reset : 해당 버튼이 포함된 폼의 입력 내용이 초기화됩니다.

Quiz #6: 화면 전체를 "lime" 색상으로 설정하려면 어느 태그에 어떤 코딩을 추가해야 할까요?
(1) body 태그 안에 적는 방법
<body style="background-color:lime">

(2) CSS 부분을 따로 style태그로 분리
<style>
    body {background-color:lime;}

(3) CSS 부분을 다른 파일로 분리
<head><link rel="stylesheet" href="1.css"></head>

1.css
@charset "UTF-8";
body{background-color:lime;}

Quiz #7: 아래 화면 처럼 나오게 코딩해보세요.
1장 확인문제 7
- 맨/워먼 중 하나만 선택할 수 있고, 식성 3개 중 하나만 선택할
수 있도록 하려면 name속성을 어떻게 설정하면 되나요?
- value속성의 값은 영어로 넣으세요.
<body><form action="" method="">
    성별: <input type="radio" name="gender" value="man">맨
        <input type="radio" name="gender" value="waman">워먼<br>
    식성: <input type="radio" name="food" value="meet">고기only
        <input type="radio" name="food" value="vat">채소only
        <input type="radio" name="food" value="all">뭐든

Quiz #8: 좌측 소스를 수정해서 우측 화면이 나오도록 하세요
<h2>HTML 확인문제</h2>
안녕하세요, 아이유입니다.
반값습니다!
<style>
    h2 {background-color:yellow; color:gray}
    strong {background-color:aqua;}
</style>
<body><h2>HTML 확인문제</h2>
    안녕하세요, <strong>아이유</strong>입니다.<br>반갑습니다.

Quiz #9: CSS 기초 문제입니다.
수업자료 1장 p. 23 소스를 아래와 같이 바꿔보세요.
(1) sony를 파란색으로 바꿔보세요. (id속성 이용)
 <td id="blue> 태그 사용 <style>태그에서 #blue 선언 
(2) sony외에 손흥민도 파란색으로 바꿔보세요.
(class 속성 이용)
<td class="blue">태그 사용 <style>태그에서 .blue 선언

Quiz #10: 첫 과제!
    (1) 좌측화면하고 최대한 똑같이 구현
    (2) 타이틀에는 본인 학번 이름
    (3) form은 넣어도 되고 안 넣어도 무관
    (4) 테이블은 4행 4열입니다. 들어있는 내용은 개성있게 바꿔보세요.
    (5) ID : 앞부분에 &nbsp;를 넣어서 ID, 이름, 학년, 내용 세로 줄 최대한 정렬되도록
    (6) type= ”text” 박스 크기 조절은 size=50 이런 식으로 속성 넣어주시면 됩니다.
    (7) 맨 마지막 라인(이상 2주차...) 진하게
    (8) 소스 (txt파일) 및 화면 (그림파일) 제출

<!DOCTYPE html>
<html>
<head>
    <title>1705817엄윤상</title>
    <style>
        body {background-color: lime;}
        table, th, td {border: 1px solid black;}
    </style>
</head>
<body>
    <hr>
    <h2>게시판</h2>
    <hr><br>
    <table>
        <tr>
            <th>ID <th>이름 <th>학년 <th>내용
        <tr>
            <td>ppiyak <td>나삐약 <td>2 <td>저는 아직도 삐약이라서 코딩 어려워요
        <tr>
            <td>great <td>엄청나 <td>4 <td>저는 갓코딩입니다. 으허허허
        <tr>
            <td>&nbsp; <td>&nbsp; <td>&nbsp; <td>&nbsp;
        </tr>
    </table>
    <p>
        &nbsp;&nbsp; ID : <input type="text"><br>
        이름 : <input type="text"><br>
        학년 :
        <input type="radio"> 1
        <input type="radio"> 2
        <input type="radio"> 3
        <input type="radio"> 4<br>
        내용 : <input type="text" size="50">
    </p>
    <strong>이상 2주차 과제였습니다 ^^</strong>
</body>
</html>

```
