# E-16 다음에 볼 영화 예측하기

총 2가지 기준으로 `SessionId`를 정의하였다.
1. `UserId`를 SessionId로 정의
2. `UserId`와 `Time`을 기준으로 SessionId를 정의
  
### 1. `UserId`를 SessionId로 정의
- 코드 :[[E-16] Movielens_SBR](  )
- 최종 하이퍼파라미터 : batch_size=128, hsz=50, drop_rate=0.1, lr=0.001, epochs=10, k=20
- 최종 성능 : Recall@20: 0.151042 / MRR@20: 0.049617

### 2. `UserId`와 `Time`을 기준으로 SessionId를 정의
- 코드 : [[E-16] Movielens_SBR_2](  )
- 최종 하이퍼파라미터 : 위와 동일하게 진행
- 최종 성능 : Recall@20: 0.247396 / MRR@20: 0.090361

#### ▶ `UserId`와 `Time`을 기준으로 SessionId를 정의한 것이 그렇지 않은 경우보다 성능이 좋은 것을 확인할 수 있다.
