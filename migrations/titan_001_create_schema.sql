-- Migration: Create TITAN OMNISCIENCE schema
-- Date: 2026-01-25
-- Purpose: FASE 1 - Enterprise scraping infrastructure with PIT compliance
-- Plan: zazzy-jingling-pudding.md v1.1

-- =============================================================================
-- TITAN OMNISCIENCE Schema
-- Sistema de Big Data para Predicción Deportiva
-- =============================================================================

-- Create isolated schema for TITAN infrastructure
CREATE SCHEMA IF NOT EXISTS titan;

-- Trigger function for automatic updated_at (reusable across titan.* tables)
CREATE OR REPLACE FUNCTION titan.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Schema documentation
COMMENT ON SCHEMA titan IS 'TITAN OMNISCIENCE: Enterprise scraping with PIT compliance. Isolated from public.* schema.';
COMMENT ON FUNCTION titan.update_updated_at_column() IS 'Auto-update updated_at timestamp on row modification';
