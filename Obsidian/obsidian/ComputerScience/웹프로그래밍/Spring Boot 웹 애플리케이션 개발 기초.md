#ComputerScience #웹프로그래밍 

---
### Spring Boot 기초

#### 1. MyController.java
	1.public class...윗 줄에 @Controller가 있어야 이 자바소소는 컨트롤러 역할을 합니다.
	2.컨트롤러 소스안에는 메소드들이 많습니다. 메소드들의 순서는 상관없습니다.
	3.이번 학기 내내 우리가 작성하는 컨트롤러의 메소드에는 @GetMapping("/주소") 혹은@PostMapping("/주소")가 붙습니다.
	4.자바에서 @로 시작하는 것은 이노테이션이라 부릅니다. 우리는 현재까지 3가지 이노테이션을 사용했습니다.
	5.GetMapping PostMapping에 들어가는 주소는 http://localhost:8080는 생략하고 슬래쉬 /부터 시작합니다. / 하나만 있으면 http://localhost:8080/라는 의미입니다.
	6.앞에서 보낸 데이터를 받으려면 메소드 괄호 안에 String mid 이런식으로만 적어도name속성이 mid인 데이터를 잘 받습니다. 다만, 앞에서 보낸 데이터임을 확실하게 못박기 위해서 @RequestParam(name="mid")를 적는다.
	7.get vs post
		get : 주소창에 모든 정보가 포함되어 표현됨
		post : 모든 정보가 숨겨짐 (get보다 보안이 좋음)
	8. get <-> post 바꾸는 법
		1. 보내는 html <form> 태그 안 action="get,post"로 변경
		2. 받는 Controller @Get,Post Mapping으로 변경
	9. 
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
    public String ex04Answer(Model mo) {
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
		th:text는 태그 속성 : 무조건 태그안에 있어해해
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
<option>임원
</select><p>
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
		변수와 하드코딩을 썩을땐 "" 안에 ||를 사용하면 됨
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
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
<meta charset="UTF-8"><title>list</title>
<link rel="stylesheet" href="/ex04.css">
</head>
<body>
<h2>회원 리스트 

