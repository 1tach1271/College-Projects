-- Saveur Restaurant OS schema for Supabase Postgres
-- Run this in Supabase SQL editor.

create table if not exists users (
  id bigint primary key,
  name text not null,
  email text,
  phone text,
  password text not null,
  role text not null check (role in ('customer', 'owner')),
  verified boolean not null default false,
  createdAt timestamptz default now()
);

create table if not exists menu_items (
  id bigint primary key,
  name text not null,
  price numeric(10,2) not null,
  cat text not null,
  emoji text not null,
  "desc" text,
  avail boolean not null default true,
  stock integer not null default 0
);

create table if not exists orders (
  id bigint primary key,
  items jsonb not null,
  total numeric(10,2) not null,
  status text not null,
  date timestamptz not null,
  cust text not null,
  custId bigint not null
);

create table if not exists feedbacks (
  id bigint primary key,
  "user" text not null,
  userId bigint not null,
  rating integer not null check (rating between 1 and 5),
  comment text,
  item text,
  date timestamptz not null
);

create table if not exists notifications (
  id bigint primary key,
  icon text not null,
  title text not null,
  text text not null,
  time text,
  type text not null,
  read boolean not null default false,
  targetRole text,
  targetUserId bigint
);

-- For a demo app, permissive policies are simplest.
-- Tighten these for production.
alter table users enable row level security;
alter table menu_items enable row level security;
alter table orders enable row level security;
alter table feedbacks enable row level security;
alter table notifications enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies where schemaname='public' and tablename='users' and policyname='public_all_users'
  ) then
    create policy public_all_users on users for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies where schemaname='public' and tablename='menu_items' and policyname='public_all_menu_items'
  ) then
    create policy public_all_menu_items on menu_items for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies where schemaname='public' and tablename='orders' and policyname='public_all_orders'
  ) then
    create policy public_all_orders on orders for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies where schemaname='public' and tablename='feedbacks' and policyname='public_all_feedbacks'
  ) then
    create policy public_all_feedbacks on feedbacks for all using (true) with check (true);
  end if;
  if not exists (
    select 1 from pg_policies where schemaname='public' and tablename='notifications' and policyname='public_all_notifications'
  ) then
    create policy public_all_notifications on notifications for all using (true) with check (true);
  end if;
end $$;
