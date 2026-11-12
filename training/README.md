# training

Colab 노트북과 데이터 변환 스크립트를 둔다. 데이터·가중치는 커밋하지 않는다.

| 파일 | 담당 | 내용 |
| --- | --- | --- |
| `data.yaml` | 공동 | YOLO 클래스 정의 (`pipeline/schema.py`의 `MaskClass - 1`) |
| `aihub_scene_extract.py` | A | AI Hub 인도보행 서피스마스킹 → 장면 모델 YOLO-seg(`0 crosswalk`, `1 road`). 영상 폴더 단위로 train/val 분리 |
| `convert_crutch.py` | A | Roboflow crutch-data → 같은 클래스 체계로 재매핑 |
| `train_scene.ipynb` | A | 장면 모델(crosswalk, road) Colab 학습. 가중치는 Drive `tactilecheck/runs/scene_v1/` |
| `finetune_cheongju.ipynb` | B | 청주 실촬영 미세조정 (A 가중치에서 시작, 없으면 공개 사전학습 가중치) |

합성 마스크 생성기는 규칙 테스트에서도 쓰므로 `pipeline/synth.py`에 있다.

가중치는 Google Drive / Hugging Face Hub / GitHub Release 중 한 곳에 올리고,
서버에서는 `weights/best.pt`로 받아 둔다.
