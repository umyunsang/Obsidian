
---
아래는 코드와 함께 설명을 포함한 내용입니다:

### Spring Boot 기초

#### 1. MyController.java
```java
package com.web.p1;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.ArrayList;

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
```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
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
```

1. **Spring Boot 기초 개념**: Spring Boot를 사용하여 웹 애플리케이션을 개발하는 기본 개념을 이해해야 합니다.
2. **Controller 작성**: `@Controller` 어노테이션을 사용하여 컨트롤러를 작성하고, 요청 매핑을 설정합니다.
3. **Thymeleaf 템플릿 엔진**: Thymeleaf를 사용하여 HTML 템플릿을 작성하고, 컨트롤러와 데이터를 연동하여 동적으로 웹 페이지를 생성합니다.
4. **POST와 GET 요청**: HTTP POST와 GET 요청의 차이를 이해하고, 각각의 요청에 대한 처리를 구현합니다.
5. **Model 객체 활용**: `Model` 객체를 사용하여 컨트롤러에서 뷰로 데이터를 전달하는 방법을 익혀야 합니다.
6. **HTML과 CSS**: HTML과 CSS를 사용하여 웹 페이지의 레이아웃과 스타일링을 설정합니다.
7. **반복문과 조건문 활용**: Thymeleaf에서 제공하는 반복문과 조건문을 사용하여 동적인 컨텐츠를 생성합니다.

이러한 내용들을 숙지하면 Spring Boot를 활용하여 간단한 웹 애플리케이션을 개발하는 데 필요한 기본적인 지식을 습득할 수 있을 것입니다.