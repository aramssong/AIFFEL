# [Aiffel Datathon] 전력사용량 예측 데이터톤
![image](https://user-images.githubusercontent.com/97036411/188098053-65fd9fd1-6854-4492-aeb3-00a6dd260f0d.png)

- **데이터** : 60개 건물들의 2020년 6월 1일 부터 2020년 8월 24일까지의 데이터
    - 1시간 단위로 제공
    - 전력사용량(kWh) 포함
    - 122,400개 데이터, 10개 컬럼
---
DACON에 있는 전력사용량 예측 AI 경진대회의 데이터를 분석하는 주제인 데이터톤의 결과물입니다.
    
([DACON 전력사용량 예측 AI 경진대회](https://dacon.io/competitions/official/235736/overview/description))

✔ 기간 : 22.03.08(화) ~ 22.03.11(금)
    
✔ 팀원 : 류정민, 박준희, 송아람 (아이펠 울산 1기)
    
---
📌 **분석 목적**
    
    각 건물별 전력 사용량 파악, 시각화하여 건물의 특성에 따라 미치는 요소를 발견해보자
    
    
📌 **분석방향**
1. 각 건물 별 전력사용량을 요일, 시간대 별로 확인
2. K-means로 전력 사용 경향에 따른 건물의 군집화 진행
3. 기상 feature(기온, 습도, 강수량, 바람, 일조량)과 전력사용량(target)의 상관관계 확인
4. Heat map으로 나타낸 각 군집 내 패턴이 다른 건물(이상치)을 확인하여 이상치로부터 얻을만한 인사이트가 있는지 확인
5. 각 군집과 기상 feature 간의 상관관계 확인
    
    
 
📌 **결론**
- **각 군집 별 feature과의 상관관계**

|  | 군집 특성 | 상관관계 |
| --- | --- | --- |
| 군집 0번 | 저녁과 주말에 전력사용량 많음 | ‘기온’과 전력사용량의 상관관계가 높음 |
| 군집 1번 | 06시 ~ 18시에 전력사용량 많음 | ‘기온’, ‘일조량’과 전력사용량의 상관관계가 높음 |
| 군집 2번 | 각 건물들의 패턴이 다름 | 확인 불가 |
| 군집 3번 | 10시 ~ 18시, 주말에 전력사용량 많음 | ‘기온’, ‘일조량’과 전력사용량의 상관관계가 높음 |
    
    
- 저녁에 전력사용량이 많은 `군집 0번`은 **일조량**의 영향을 많이 받지 않음
- 오전, 오후에 전력사용량이 많은 `군집 1번`, `군집 3번`은 **기온과 일조량**의 영향을 많이 받음
    
    
- 4개의 군집 모두 `비전기냉방시설`, `태양열`과의 상관관계가 낮음
    → 이러한 장치는 전력사용량에 영향을 미치지 않는 것으로 확인.

