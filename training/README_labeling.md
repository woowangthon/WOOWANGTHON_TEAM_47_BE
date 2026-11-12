# AI Hub 점자블럭 라벨링 가이드

AI Hub 점자블럭 사진을 YOLO-seg 라벨(`0 linear`, `1 dot`)로 만드는 순서를 정리함.
명령은 모두 레포 최상위 폴더에서 실행함.

## 1. 후보 뽑기

원본 zip(`_data/`)은 풀지 않고 필요한 사진만 꺼냄. 이미 뽑은 사진은 `--exclude`로 빼서 겹치지 않게 함.

```bash
.venv/bin/python training/aihub_extract.py --n 500 \
  --exclude datasets/aihub_used_batch1.txt --exclude datasets/aihub_used_batch2.txt \
  --used-out datasets/aihub_used_batch3.txt --out datasets/aihub_candidates_3
```

- 다음 묶음은 `--exclude`에 `batch3`까지 넣고 번호를 하나씩 올림
- 결과: `images/`(1280px), `json/`(AI Hub 원본), `gallery.html`(미리보기)

## 2. 폴더 분류

- `gallery.html`로 훑어보고 라벨링할 사진을 새 폴더로 옮김 (예: `점형 및 혼합/`)
- 선형은 이미 충분하므로 점형·혼합 위주로 고름
- 옮긴 사진의 JSON도 같은 폴더로 옮겨야 함 (사진과 JSON이 나란히 있어야 함)

## 3. 앱 입력 준비

```bash
.venv/bin/python training/prepare_aihub_pilot.py \
  --images "datasets/aihub_candidates_3/점형 및 혼합" --json "datasets/aihub_candidates_3/점형 및 혼합" \
  --out "datasets/aihub_pilot/batch3_점형 및 혼합"
```

## 4. 라벨링 앱 실행

```bash
.venv/bin/python training/label_aihub_extra.py \
  --input "datasets/aihub_pilot/batch3_점형 및 혼합" --out "datasets/aihub_labeled/batch3_점형 및 혼합"
```

- 입력기는 **영문**으로 둠 (한글이면 키가 안 먹음)
- 주황 테두리 = AI Hub가 그린 블록 영역

### 키

| 키 | 기능 |
| --- | --- |
| `L` / `D` | 주황 테두리 **밖** 선형 / 점형 블록 추가. 클릭 → `Enter` |
| `1` / `2` | 주황 영역 전부 선형 / 전부 점형으로 저장 |
| `3` → 클릭 → `Enter` → `S` | 주황 영역이 섞였을 때. 점형 부분만 그림 (나머지는 자동으로 선형) |
| 오른쪽 클릭, `Backspace` | 그리던 점 하나 지움 |
| `Z` | 완성한 다각형 지움 |
| `X` | 제외 |
| `←` / `→` | 이전 / 다음 사진 |
| `E` → `Q` | 내보내기 후 종료 (**`E` 먼저**) |

### 사진 한 장 처리 순서

1. 테두리 밖에 빠진 블록이 있으면 `L`/`D`로 먼저 추가함
2. 주황 영역을 `1` / `2` / `3`(+`S`) 중 하나로 저장함. 추가한 블록은 같이 저장됨

### 규칙

- 테두리 **밖** 블록은 `L`/`D`, 테두리 **안**을 나누는 건 `3`
- `S`는 혼합 저장 전용. 주황 영역이 한 종류면 `1`/`2`를 누름
- 다각형은 블록 줄을 끝에서 끝까지 가로지르게, 주황 선 바깥까지 넉넉하게 그림 (안쪽에 떠 있으면 "구멍" 오류)
- 추가 블록이 주황 영역과 겹치면 겹친 부분은 추가 블록 클래스를 따름. 추가 선형·점형끼리 겹치면 선형 우선
- 판단이 어려우면 `X`

## 5. 결과

- `datasets/aihub_labeled/<폴더>/images/all`, `labels/all`에 저장됨
- 중간에 닫아도 결정은 `quick_label_progress.json`에 남음. 다시 열고 `E`만 누르면 됨
- 학습용 묶음(`datasets/aihub_yolo_v2/`)에 합칠 때는 사진·라벨 짝과 클래스 번호(0, 1만)를 확인함
