-- TactileCheck 점검 결과 테이블. Supabase 대시보드 → SQL Editor 에서 한 번 실행한다.
-- 서버(FastAPI)가 service_role 키로 읽고 쓴다. 프론트는 Supabase에 직접 접근하지 않고 API만 부른다.

create table if not exists public.inspections (
  id             uuid primary key,
  created_at     timestamptz not null default now(),
  overall        text not null check (overall in ('suspected', 'normal', 'hold')),
  schema_version text not null,
  result         jsonb not null            -- InspectionResult 전체 (판정 목록, scope, 분석 결과)
);

create index if not exists inspections_created_at_idx on public.inspections (created_at desc);

-- 공개(anon) 접근을 막는다. service_role 키는 RLS를 우회하므로 서버 동작에는 영향 없음.
alter table public.inspections enable row level security;

-- 프로젝트를 'Automatically expose new tables' 끄고 만들었으면 서버 역할에 권한을 직접 준다.
-- (켜고 만들었어도 실행해도 문제없음. anon·authenticated에는 주지 않는다)
grant select, insert, update on public.inspections to service_role;
