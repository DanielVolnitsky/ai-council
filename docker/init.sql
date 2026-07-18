-- docker/init.sql
--
-- Executed once by the postgres Docker entrypoint when the container is first
-- created (files in /docker-entrypoint-initdb.d/ are run in sorted order).
--
-- Creates the council_test database used by the test suite.
-- The main council database is created by the POSTGRES_DB env var in
-- docker-compose.yml, so we only need to add the test database here.

CREATE DATABASE council_test;
