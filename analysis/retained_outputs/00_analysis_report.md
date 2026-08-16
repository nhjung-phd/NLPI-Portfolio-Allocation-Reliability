# NLPI Q2 통계분석 자동 보고서

- 실행 시각: 2026-08-01 12:01:18.591298
- 프로젝트: `/Users/nhjung/Data/git/LLM_Portfolios/LLM_Portfolio_NLPI_PaperCanonical`
- 통합 신뢰성 표본: **5,670건** (기대값 5,670건과 일치)
- 최초 parse failure: **2건**
- repair 사용: **19건**
- 최종 JSON validity: **100.00%**
- cross-family 정책 수: **6개**
- Study B P2 projected fidelity 평균: **0.0000**

## 해석 원칙

1. 5,670건 전체를 독립 관측치로 해석하지 않는다. Friedman 검정은 공통 날짜를 block으로 사용했다.
2. 유의한 omnibus 결과에만 Wilcoxon signed-rank 사후검정을 적용하고 Holm 보정을 보고한다.
3. all-zero 대응차는 검정하지 않고 Holm 보정에서도 제외한다.
4. GEE는 실험별로 적합하며, 분리가 남는 경우 군집강건 선형확률 GEE를 명시적으로 사용한다.
5. P2 기준정책은 Study A/B에서 동일하지만 실행 문맥의 단절이 남아 두 결과를 비교·통합하려면 72건 bridge validation을 권고한다.
6. 포트폴리오 검정은 중복 fold overlay를 제외하고 stitched OOS equity에서 모델×페르소나별 일수익률을 복원한다.

## 생성 결과

CSV 표 16개와 그림 2개를 생성했다.
