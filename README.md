# TactileCheck BE

스마트폰 사진 두 장(근접 + 넓은)으로 점자블록 설치 오류를 찾아
**설치 오류 의심 / 정상 추정 / 판정 보류**로 알려주는 서비스의 백엔드와 판정 파이프라인.

## 구조

```
app/                FastAPI 서버, 결과 저장 (A)
pipeline/
  schema.py         A↔B 접점: 마스크·경로 구조·판정 결과 형식 (11/10 합의본)
  segment.py        ② YOLOv8-seg 추론 → 라벨 마스크
  skeleton.py       ③ 중심선 → 끝점·꺾임점·분기점 그래프 (A)
  rectify.py        ④ 시점 보정, ROI 돌출선 방향 FFT (B)
  rules/            ⑤ 규칙 판정
    m_rules.py      M3·M4 (A)
    c_rules.py      C1~C4 방향·연결 판정 (B)
    hold.py         H1·H4 (A), H2·H3 (B)
    judge.py        판정 순서와 종합 판정
    clauses.py      근거 조항·개선 방향 (공동)
  synth.py          합성 라벨 마스크 생성기 (B)
  run.py            파이프라인 진입점
training/           데이터 변환 스크립트, Colab 노트북, data.yaml
tests/              pytest (합성 마스크 기반 규칙 테스트)
docs/               JSON 예시
weights/            모델 가중치 (커밋하지 않음)
```

## 라벨 마스크 형식

1채널 uint8 PNG. 픽셀 값은 `pipeline/schema.py`의 `MaskClass`를 따른다.

| 값 | 클래스 | YOLO id |
| --- | --- | --- |
| 0 | 배경 | – |
| 1 | 선형 블록 | 0 |
| 2 | 점형 블록 | 1 |
| 3 | 횡단보도 | 2 |
| 4 | 차도 | 3 |

경로 구조와 판정 결과 예시는 [docs/path_structure.example.json](docs/path_structure.example.json),
[docs/inspection_result.example.json](docs/inspection_result.example.json)을 보면 된다.
형식을 바꾸면 `SCHEMA_VERSION`을 올리고 FE에도 알린다.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # 서버 + 테스트
pip install -r requirements-ml.txt       # 모델 추론까지 할 때
cp .env.example .env
pytest
uvicorn app.main:app --reload
```

모델 없이 정답 마스크로 개발하기:

```python
import cv2
from pipeline.run import analyze_label
from pipeline.rules import judge
from pipeline.schema import PhotoKind

close = analyze_label(cv2.imread("close_mask.png", 0), PhotoKind.CLOSE)
wide_label = cv2.imread("wide_mask.png", 0)
wide = analyze_label(wide_label, PhotoKind.WIDE)
print(judge(close, wide, wide_label).model_dump_json(indent=2))
```

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/api/health` | 상태, 스키마 버전, 모델 로드 여부 |
| POST | `/api/inspections` | multipart `close`, `wide` 사진 → `InspectionResult` |
| GET | `/api/inspections/{id}` | 저장된 결과 조회 |

## 커밋하지 않는 것

- AI Hub 데이터 (재배포 금지), 청주 촬영 원본 (얼굴·번호판), 모델 가중치 — `.gitignore`에 등록됨
