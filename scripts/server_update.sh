#!/usr/bin/env bash
# EC2 서버에서 최신 코드로 다시 배포한다. 서버의 ~/tactilecheck 에서 실행:  bash scripts/server_update.sh
# (.env는 서버에만 있고 git이 건드리지 않는다)
set -euo pipefail
cd "$(dirname "$0")/.."
git pull --ff-only
docker build -t tactilecheck-api .
docker rm -f api 2>/dev/null || true
docker run -d --name api --restart unless-stopped --env-file .env -p 127.0.0.1:7860:7860 tactilecheck-api
echo "모델 로드 대기..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:7860/api/health/deep >/dev/null; then
    curl -s http://127.0.0.1:7860/api/health/deep; echo; echo "배포 완료"; docker image prune -f >/dev/null; exit 0
  fi
  sleep 2
done
echo "60초 안에 응답 없음 → docker logs api --tail 50 확인"; exit 1
