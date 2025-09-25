# 아키텍처 다이어그램 (텍스트 버전)

```
┌─────────────────────────────────────────────────────────────────┐
│                    소셜 비디오 빅데이터 실시간 처리 플랫폼                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Sources   │    │   Ingest     │    │   Bronze    │    │   Silver    │
├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤
│YouTube API  │───▶│Reservoir    │───▶│Bronze Delta │───▶│Structured   │
│Social Events│    │Sampling     │    │(partitioned)│    │Streaming    │
└─────────────┘    │Bloom Filter │    └─────────────┘    │Watermark    │
                   │Landing JSON │                       │Sliding Win  │
                   └─────────────┘                       └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Gold      │    │  Modeling   │    │  Serving    │    │  Checkpoint │
├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤
│HLL/approx   │◀───│LR/RF/GBT    │◀───│Streamlit   │    │chk/silver/  │
│CDF cutoff   │    │Pareto Front │    │UI           │    │             │
│Gold Features│    │Artifacts    │    │Predictions  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 데이터 흐름

1. **Sources** → **Ingest**: YouTube API와 소셜 이벤트 데이터 수집
2. **Ingest** → **Bronze**: Reservoir Sampling과 Bloom Filter를 통한 중복 제거
3. **Bronze** → **Silver**: Structured Streaming과 Watermark를 통한 실시간 처리
4. **Silver** → **Gold**: HLL과 CDF를 통한 피처 엔지니어링
5. **Gold** → **Modeling**: 다중 모델 학습 및 Pareto Front 최적화
6. **Modeling** → **Serving**: Streamlit UI를 통한 결과 제공
