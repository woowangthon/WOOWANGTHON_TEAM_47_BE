# TactileCheck API (프론트 연동용)

| | 주소 |
| --- | --- |
| **배포 (EC2, HTTPS)** | `https://3-107-144-61.sslip.io` |
| 로컬 개발 | `http://localhost:8000` (`uvicorn app.main:app --port 8000`) |

스키마 버전 `0.3.2`. 브라우저에서 `https://3-107-144-61.sslip.io/docs`를 열면 모든 API를 직접 눌러 볼 수 있다.

- 프론트 배포 주소(예: Vercel)는 서버 CORS 허용 목록에 넣어야 브라우저 요청이 통과한다 → 주소가 정해지면 백엔드 담당에게 알려 줄 것
- 위치(`navigator.geolocation`)는 HTTPS 페이지에서만 동작한다 (localhost는 예외)

## 시민 화면

### 점검(신고) 접수 — `POST /api/inspections`
`multipart/form-data`

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `photo` | ✅ | 사진 한 장 (JPG·PNG). **최대 15MB.** 폰 사진은 업로드 전에 긴 변 2000px 정도로 줄이면 빠르다 |
| `address` | | 주소 또는 장소 이름 (최대 200자) |
| `memo` | | 장소 설명 (예: "우암초 앞 횡단보도", 최대 1000자) |
| `lat`, `lng` | | 위도·경도. 브라우저 `navigator.geolocation`으로 받은 값 |

응답 `200`: InspectionResult (아래). 오류: `400` 사진 없음·읽기 실패, `413` 너무 큼, `422` 위치 값 오류,
`429` 같은 IP에서 분당 10회 초과, `503` 서버 모델 준비 안 됨.

```js
const fd = new FormData();
fd.append("photo", file);              // 줄인 Blob이면 fd.append("photo", blob, "photo.jpg")
fd.append("address", address);         // 빈칸이면 안 보내도 됨
if (coords) { fd.append("lat", coords.latitude); fd.append("lng", coords.longitude); }
const res = await fetch(`${API}/api/inspections`, { method: "POST", body: fd });
const result = await res.json();
```

### 결과 다시 보기 — `GET /api/inspections/{id}`
접수 때와 같은 InspectionResult. 결과 페이지 주소에 `id`를 넣어 두면 새로고침·공유해도 다시 불러올 수 있다.

### 신고 접수 — `POST /api/inspections/{id}/report`
결과 화면에서 "신고하기"를 누른 뒤 발견 위치·내용을 보낸다. JSON 본문:

```json
{"address": "청주시 우암동 우암초 앞 횡단보도", "memo": "선형 블록이 횡단보도로 이어지지 않음", "lat": 36.65, "lng": 127.49}
```

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `address` | ✅ | 발견 위치 (최대 200자, 빈칸 불가) |
| `memo` | | 확인이 필요한 내용 (최대 1000자) |
| `lat`, `lng` | | 위도·경도 (선택) |

응답: 관리자 목록 한 줄과 같은 형식 (`inspection_id, status, codes, location …`). 관리자 신고 목록에 이 위치가 나온다.
처리 상태가 `received`(접수)인 동안은 다시 보내면 덮어쓴다.
오류: `404` 없는 점검, `409` 관리자가 이미 확인 중, `422` 값 오류, `502` 저장 실패(접수되지 않음 → 다시 시도 안내).

### 사진 — `GET /api/inspections/{id}/photo`
저장된 사진 (긴 변 1600px JPEG). `<img src={API + result.photo_url} />`로 바로 쓴다.

## InspectionResult (주요 필드)

| 필드 | 화면 사용 |
| --- | --- |
| `inspection_id` | 결과·관리자 페이지 주소 |
| `overall` | `suspected` 설치 오류 의심(빨강) · `normal` 정상 추정(초록) · `hold` 판정 보류(회색) |
| `findings[]` | 판정 카드: `code`, `verdict`, `message`, `location {x, y}`, `clause`(근거 조항, null이면 "확인 중"), `suggestion`(개선 방향), `primary`(false면 "보조") |
| `scope` | `{"C": "판정", "M": "건너뜀 (…)"}` — 어떤 판정을 했는지 |
| `photo_url` | 사진 주소 (위) |
| `location` | `{address, memo, lat, lng}` (없으면 null) |
| `created_at` | 접수 시각 (ISO 8601, UTC) |
| `photo.segmentation.width/height` | **판정 위치 좌표의 기준 크기** (아래) |

### 사진 위에 판정 위치 핀 찍기
`findings[].location`은 **원본 사진 픽셀 좌표**다. 화면에 보이는 사진 크기에 맞춰 비율로 바꾼다.

```js
const { width, height } = result.photo.segmentation;      // 원본 크기
const sx = imgEl.clientWidth / width, sy = imgEl.clientHeight / height;
const pins = result.findings
  .filter(f => f.location)
  .map((f, i) => ({ n: i + 1, left: f.location.x * sx, top: f.location.y * sy, verdict: f.verdict }));
```
저장된 사진은 줄여서 저장되지만 가로세로 비율은 같으므로 위 계산 그대로 쓰면 된다.

### 보류 판정 표시
`verdict: "hold"`인 판정은 회색 카드로, `message`에 이유와 재촬영 안내가 들어 있다
(예: "블록 모델이 읽지 못한 점자블록이 있어 판정 보류. 블록이 더 크게 나오게 찍어 주세요 (원래 판정: …)").
`clause`·`suggestion`은 null일 수 있다.

### 판정 코드 화면 이름
M3 꺾임·분기점 점형 누락 · M4 선형 경로 끊김 · C1 횡단보도 연결 누락 · C2 차도 방향 유도(위험) ·
C3 꺾임 안내 누락 · C4 진입부 점형·방향 오류 · C5 선형 유도 없음 · C6 선형 방향 불일치 · H1~H4 판정 보류

## 관리자 화면 (요청 헤더 `X-Admin-Token: <토큰>` 필요)

| API | 설명 |
| --- | --- |
| `GET /api/admin/inspections?status=&overall=&limit=50&offset=0` | 목록 (최근 접수 순). 항목: `inspection_id, created_at, overall, status, codes[], location, photo_url` |
| `GET /api/admin/inspections/{id}` | 상세: 위 항목 + `admin_memo, updated_at, result`(InspectionResult 전체) |
| `PATCH /api/admin/inspections/{id}` | JSON `{"status": "reviewing", "admin_memo": "현장 확인"}` (둘 중 하나만 보내도 됨) |

처리 상태 `status`: `received` 접수 · `reviewing` 확인 중 · `resolved` 처리 완료 · `rejected` 반려.
오류: `401` 토큰 없음·틀림, `404` 없는 신고, `503` 서버에 관리자 토큰 미설정.

```js
const res = await fetch(`${API}/api/admin/inspections?status=received`, {
  headers: { "X-Admin-Token": adminToken },   // 관리자가 화면에서 입력, 코드에 박아 두지 말 것
});
```

## 서버 상태
- `GET /api/health` — `{ok, schema_version, model_loaded, storage}`
- `GET /api/health/deep` — 모델 + 저장소(Supabase) 연결까지 확인. 모니터링(UptimeRobot, 6시간 간격)용
