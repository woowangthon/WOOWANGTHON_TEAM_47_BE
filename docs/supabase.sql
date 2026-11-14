-- TactileCheck 점검(신고) 테이블 + 사진 버킷. Supabase 대시보드 → SQL Editor 에서 실행한다.
-- 처음 만들 때도, 이미 만든 테이블을 최신으로 바꿀 때도 이 파일 전체를 그대로 실행하면 된다 (여러 번 실행해도 안전).
-- 서버(FastAPI)가 secret(service_role) 키로 읽고 쓴다. 프론트는 Supabase에 직접 접근하지 않고 API만 부른다.

-- 1. 테이블 ------------------------------------------------------------------------------------------
create table if not exists public.inspections (
  id             uuid primary key,
  created_at     timestamptz not null default now(),
  overall        text not null check (overall in ('suspected', 'normal', 'hold')),
  schema_version text not null,
  result         jsonb not null            -- InspectionResult 전체 (판정 목록, scope, 분석 결과)
);

-- 0.3.0: 사진·위치·처리 상태 (이미 있으면 건너뜀)
alter table public.inspections add column if not exists photo_path text;                 -- Storage 안 경로
alter table public.inspections add column if not exists codes      text[] not null default '{}';  -- '오류 의심' 코드
alter table public.inspections add column if not exists address    text;                 -- 주소·장소 이름
alter table public.inspections add column if not exists place_memo text;                 -- 장소 설명
alter table public.inspections add column if not exists lat        double precision;
alter table public.inspections add column if not exists lng        double precision;
alter table public.inspections add column if not exists status     text not null default 'received';
alter table public.inspections add column if not exists admin_memo text;
alter table public.inspections add column if not exists updated_at timestamptz not null default now();

do $$ begin
  alter table public.inspections add constraint inspections_status_check
    check (status in ('received', 'reviewing', 'resolved', 'rejected'));
exception when duplicate_object then null; end $$;

create index if not exists inspections_created_at_idx on public.inspections (created_at desc);
create index if not exists inspections_status_idx on public.inspections (status, created_at desc);

-- 2. 접근 권한 ---------------------------------------------------------------------------------------
-- 공개(anon) 접근을 막는다. service_role 키는 RLS를 우회하므로 서버 동작에는 영향 없음.
alter table public.inspections enable row level security;
-- 'Automatically expose new tables'를 끄고 만든 프로젝트용. 서버 역할에만 준다 (삭제 권한은 주지 않음).
grant select, insert, update on public.inspections to service_role;

-- 3. 사진 버킷 (비공개) -------------------------------------------------------------------------------
-- 주소를 알아도 볼 수 없다. 사진은 서버 API(/api/inspections/{id}/photo)로만 내보낸다.
insert into storage.buckets (id, name, public)
values ('inspection-photos', 'inspection-photos', false)
on conflict (id) do nothing;
