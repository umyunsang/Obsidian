
---
#### openai API  활용
    
>[!<문제1>]
>post 요청 endpoint url [https://api.openai.com/v1/embeddings](https://api.openai.com/v1/embeddings) 임베딩 요청을 합니다 
>- 로컬의 엑셀파일(naver_news.xlsx)에서 0번 행의 제목 셀을 임베딩 요청하고 응답 임베딩 결과를 출력하는 코드를 구현하시오
```python
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

## openai apikey 입력
OPENAI_API_KEY = "YOUR_API_KEY"

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ").replace("\u00A0", " ")  # Non-breaking Space 제거
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + OPENAI_API_KEY
    }
    json_data = {
        'input': text,
        'model': model,
    }
    try:
        response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=json_data)
        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            return embedding
        else:
            print(f"임베딩 가져오기 오류: HTTP {response.status_code} - {response.text}")
            return None
    except Exception as e:
        # 다른 예외 처리
        print(f"임베딩 가져오기 오류: {e}")
        return None

# 데이터프레임 로드
df = pd.read_excel('./data/naver_news.xlsx')

# 첫 번째 행의 제목 추출
sample_text = df.loc[0, '제목']
print("Sample Text:", sample_text)

# 임베딩 추출
embedding = get_embedding(sample_text)
print("Embedding:", embedding)
```

---
>[!<문제2>]
>openai  SDK를 사용해서 말하는 고양이에 대한 짧은 이야기를  창작한 결과를 출력하는 코드를 구현하시오
```python
import openai
api_key = OPENAI_API_KEY

# OpenAI API 클라이언트 초기화
openai.api_key = api_key

response = openai.chat.completions.create(
     model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 창작 작가 입니다."},
        {"role": "user", "content": "말하는 고양이에 대한 짧은 이야기를 작성하시오"}
    ],
    temperature=0.8,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=1.0,
    presence_penalty=1.0
)

print("Assistant Response:")
print(response.choices[0].message)
```

>[!<문제3>]
>openai  SDK를 사용해서 로컬에서 nomuhun.txt파일을 읽어서 3문장으로 요약결과를 출력하는 코드를 구현하시오
```python
with open('./data/nomuhun.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()


response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Generate an executive summary focusing on key points and actionable insights."},
        {"role": "user", "content": f"문서를 3문장으로 요약해주세요: {file_content}"}
    ],
    temperature=0.2,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
) 

print("Original Text:\n", file_content)
print("Summary Text:\n", response.choices[0].message.content)
```

>[!<문제4>]
>사용자로부터 나이 , 성별, 선호 장르, 한국 영화 또는 외국영화, 감정상태를 입력받아서 prompt로 생성한 후 ChatGPT 영화추천을 요청하고 결과를 출력하는 코드를 구현하시오

```python
def get_movie_recommendation(age, gender, genre, country, mood) :
    prompt = f"""
    당신은 영화 추천 전문가입니다. 아래 사용자의 정보를 기반으로 개인 맞춤형 영화 5편을 추천해주세요
    각 영화의 제목, 개봉 연도, 감독, 주연배우, 간단한 줄거리를 포함해주세요
    - 나이 : {age}
    - 성별 : {gender}
    - 선호 장르 : {genre}
    - 한국영화 또는 외국영화 : {country}
    - 감정 상태 : {mood}
    
    개인 맞춤형 영화 추천해주세요"""

    try :
        response = openai.chat.completions.create(
            model ="gpt-4o-mini",
            messages =[ { "role": "system", "content" : "당신은 영화 추천 전문가입니다"},
                        { "role" : "user", "content" : prompt}  ]
        )
        recommendations = response.choices[0].message.content
        return recommendations
    except  openai.error.OpenAIError as e :
        return f"오류가 발생했습니다 : {str(e)}"       
```
```bash
age = input("나이를 입력하세요")
gender = input("성별 입력하세요 (예: 남성, 여성)")
genre = input("선호하는 장르를 입력하세요(예 : 액션, 코미디, 로맨스)")
country = input("어느 나라 영화를 선호하시나요? (예: 한국, 미국, 중국, 일본)")
mood = input("현재 감정상태를 입력하세요(예: 행복, 우울, ")
recommendations = get_movie_recommendation(age, gender, genre, country, mood)
print("**** 개인 맞춤 영화 추천 리스트 ****")
print(recommendations)
```
