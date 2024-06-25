#ComputerScience #웹프로그래밍 

---
우리가 3장에서 배운 내용의 핵심은 자료 보내기 받기!!
보내는 쪽에서 자료 받는 쪽을 정확히 명시 다음 화면도 정해졌음

그런데, 쇼핑을 한다고 생각해보면 화면 순서 미리 알 수 없음

즉, 화면끼리 넘기기만 하는 게 아니라 제 3의 장소(장바구니)에 넣어놨다가 나중에 필요할 때 꺼내 씀
#### 1. MyController.java

1. **@Controller 어노테이션**: 
   - 자바 클래스의 맨 위에 `@Controller` 어노테이션이 있어야 해당 클래스가 컨트롤러의 역할을 수행합니다.

2. **컨트롤러 메소드 순서**:
   - 컨트롤러 내부에는 여러 메소드가 있지만, 이들의 순서는 상관없습니다.

3. **컨트롤러 메소드 어노테이션**:
   - 이번 학기에 작성하는 모든 컨트롤러 메소드에는 `@GetMapping("/주소")` 또는 `@PostMapping("/주소")` 어노테이션이 필요합니다.

4. **이노테이션 사용**:
   - 자바에서 `@`로 시작하는 것을 이노테이션이라고 부르며, 현재까지 3가지 이노테이션을 사용하였습니다.

5. **GetMapping과 PostMapping 주소**:
   - `GetMapping`과 `PostMapping`에 들어가는 주소는 `http://localhost:8080`를 생략하고 슬래시(`/`)부터 시작합니다. 하나의 슬래시(`/`)만 있어도 `http://localhost:8080/`를 의미합니다.

6. **데이터 수신 메소드 작성**:
   - 앞에서 보낸 데이터를 받으려면 메소드 괄호 안에 `String mid`와 같이 작성해도 됩니다. 다만, 데이터가 앞에서 보낸 것임을 명시하기 위해 `@RequestParam(name="mid")`를 추가하는 것이 좋습니다.

7. **GET vs POST**:
   - GET은 주소창에 모든 정보가 표시되지만, POST는 모든 정보가 숨겨집니다. 따라서 POST가 GET보다 보안이 좋습니다.

8. **GET <-> POST 변경 방법**:
   - 보내는 HTML `<form>` 태그의 action 속성을 `"get"` 또는 `"post"`로 변경합니다.
   - 받는 컨트롤러의 어노테이션을 `@Get` 또는 `@Post` Mapping으로 변경합니다.
```java
@Controller
public class MyController {
    @GetMapping("/")
    public String home() {
        return "home"; // home.html을 반환
    }

    @GetMapping("/ex01")
    public String ex01() {
        return "ex01"; // ex01.html을 반환
    }

    @PostMapping("/ex01/answer")
    public String ex01Answer(@RequestParam(name="mid") String mid, Model mo) {
        mo.addAttribute("mid", mid);
        return "ex01Answer"; // ex01Answer.html을 반환
    }

    // ex02, ex02Answer 메소드 추가
    @GetMapping("/ex02")
    public String ex02() {
        return "ex02"; // ex02.html을 반환
    }

    @PostMapping("/ex02/answer")
    public String ex02Answer(@RequestParam("mname") String mname,
                             @RequestParam("po") String po, Model mo) {
        mo.addAttribute("mname", mname);
        mo.addAttribute("po", po);
        int salary = 0;
        switch(po){
            case "사원" -> salary = 3500;
            case "대리" -> salary = 5000;
            case "팀장" -> salary = 7000;
            case "임원" -> salary = 9900;
        }
        mo.addAttribute("salary", salary);
        return "ex02Answer"; // ex02Answer.html을 반환
    }

    // ex03, ex03Answer 메소드 추가
    @GetMapping("/ex03")
    public String ex03() {
        return "ex03"; // ex03.html을 반환
    }

    @PostMapping("/ex03/answer")
    public String ex03Answer(@RequestParam("mname") String mname,
                             @RequestParam("color") String color, Model mo) {
        mo.addAttribute("mname", mname);
        mo.addAttribute("color", color);
        return "ex03Answer"; // ex03Answer.html을 반환
    }

    // ex04 메소드 추가
    @GetMapping("/ex04")
    public String ex04(Model mo) {
        var arr = new ArrayList<String>();
        arr.add("고흐");
        arr.add("james");
        arr.add("dooli");
        arr.add("bread");
        // 지금은 회원정보 하드코딩. 나중에는 database에서 가져옴
        mo.addAttribute("arr",arr);
        return "ex04"; // ex04.html을 반환
    }
}
```

#### 2. home.html
```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Welcome</title>
</head>
<body style="background-color:lime">
<h2>Dooli의 p1 Project입니다!!</h2>
<a href="/ex01">예제1</a> <br>
<a href="/ex02">예제2</a> <br>
<a href="/ex03">예제3</a> <br>
<a href="/ex04">예제4</a> <br>
</body>
</html>
```

#### 3. ex01.html
```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>보내는 쪽</title>
</head>
<body style="background-color:yellow">
<form method="post" action="/ex01/answer" >
아이디: <input type="text" name="mid"> <p>
 <input type="submit" value="로그인">
</form>
</body>
</html>
```

#### 4. ex01Answer.html
	1. <strong th:text="${변수}">
		th:text는 태그 속성 : 무조건 태그안에 있어해
```html
<!DOCTYPE html>
<!-- <html xmlns:th="http://www.thymeleaf.org"> 시험에선 안써도 됨 -->
<html>
<head><meta charset="UTF-8">
<title>받는 쪽</title></head>
<body style="background-color:aqua">
<strong th:text="${mid}">james</strong>님, 반갑습니다!!
</body>
</html>
```

#### 5. ex02.html
```html
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<title>직급선택</title></head>
<body>
<form method="post" action="/ex02/answer">
이름 : <input type="text" name="mname"><p>
직급 : <select name="po">
		<option>사원
		<option>대리
		<option>팀장
		<option>임원 </select><p>
 <input type ="submit" value="확인">
</form >
</body></html>
```

#### 6. ex02Answer.html
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head><meta charset="UTF-8">
<title>연봉</title></head>
<body style="background-color:#F5ECCE">
<span th:text="${mname}">mname</span>
<span th:text="${po}">po</span>님의 연봉은
<p>
<span th:text="${salary}">salary</span> 만원
입니다.
</body></html>
```

#### 7. ex03.html
	1.<option value="aqua">시원한 아쿠아
		전송할 데이터와 화면에 보이는 데이터를 분리

```html
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<title>색 선택</title></head>
<body>
<form method="post" action="/ex03/answer">
이름 : <input type="text" name="mname"><p>
좋아하는 색 : <select name="color">
			<option value="aqua">시원한 아쿠아
			<option value="lime">라임색이 좋아요!
			<option value="orange">상큼한 오렌지색
			<option value="white">역시 흰색 최고
</select ><p>
<input type ="submit" value="보내기">
</form >
</body></html>
```

#### 8. ex03Answer.html
	1.<title th:text="|${변수}+하드코딩|">
		변수와 하드코딩을 섞을땐 "" 안에 ||를 사용하면 됨
	2.<body th:style="|background-color: ${변수}">
		이하 내용 동일
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head><meta charset="UTF-8">
<title th:text="|${mname}'s color|">
</title></head>
<body th:style="|background-color: ${color}|">
<strong th:text="${mname}">mname</strong>
님이 좋아하는 색은 <br>
<strong
th:text="${color}">color</strong>입니다.
</body>
</html>
```

#### 9. ex04.html
1. th:each -> 반복문
2. th:each="a:${arr}"> : arr 리스트 인덱스 0부터 a 변수에 선언
3. th:text="${a}"> : a 변수 출력
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
<meta charset="UTF-8"><title>list</title>
<link rel="stylesheet" href="/ex04.css">
</head>
<body><h2>회원 리스트</h2>
<table>
<tr id ="ftr"> <th> 회원 ID
<tr th:each="a:${arr}"> <td th:text="${a}">
```
#### 10. ex04.css
```css
@charset "UTF-8";
body {background-color:#ECE0F8;}
table {width:200px; border:1px dashe green;}
#ftr {background-color:orange;}
td {color:navy; text-align:center;}
```