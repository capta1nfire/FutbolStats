"""Tests for TITAN idempotency mechanism.

Verifies:
1. Idempotency key computation is deterministic
2. Same params always produce same key
3. Different params produce different keys
4. Duplicate extractions are prevented
"""

import pytest
from datetime import date
from uuid import uuid4

from app.titan.extractors.base import (
    compute_idempotency_key,
    compute_params_hash,
    ExtractionResult,
)


class TestIdempotencyKeyComputation:
    """Test idempotency key computation."""

    def test_deterministic_same_inputs(self):
        """Same inputs always produce same key."""
        key1 = compute_idempotency_key(
            source_id="api_football",
            endpoint="odds",
            params={"fixture": 12345, "bookmaker": 8},
            date_bucket=date(2026, 1, 25),
        )
        key2 = compute_idempotency_key(
            source_id="api_football",
            endpoint="odds",
            params={"fixture": 12345, "bookmaker": 8},
            date_bucket=date(2026, 1, 25),
        )
        assert key1 == key2
        assert len(key1) == 32  # CHAR(32)

    def test_different_params_different_keys(self):
        """Different params produce different keys."""
        key1 = compute_idempotency_key(
            source_id="api_football",
            endpoint="odds",
            params={"fixture": 12345},
            date_bucket=date(2026, 1, 25),
        )
        key2 = compute_idempotency_key(
            source_id="api_football",
            endpoint="odds",
            params={"fixture": 12346},  # Different fixture
            date_bucket=date(2026, 1, 25),
        )
        assert key1 != key2

    def test_different_date_bucket_different_keys(self):
        """Different date_bucket produces different key."""
        key1 = compute_idempotency_key(
            source_id="api_football",
            endpoint="odds",
            params={"fixture": 12345},
            date_bucket=date(2026, 1, 25),
        )
        key2 = compute_idempotency_key(
            source_id="api_football",
            endpoint="odds",
            params={"fixture": 12345},
            date_bucket=date(2026, 1, 26),  # Different date
        )
        assert key1 != key2

    def test_different_source_different_keys(self):
        """Different source_id produces different key."""
        key1 = compute_idempotency_key(
            source_id="api_football",
            endpoint="fixtures",
            params={"id": 12345},
            date_bucket=date(2026, 1, 25),
        )
        key2 = compute_idempotency_key(
            source_id="understat",  # Different source
            endpoint="fixtures",
            params={"id": 12345},
            date_bucket=date(2026, 1, 25),
        )
        assert key1 != key2

    def test_param_order_independent(self):
        """Param order doesn't affect key (sorted internally)."""
        key1 = compute_idempotency_key(
            source_id="api_football",
            endpoint="fixtures",
            params={"league": 140, "season": 2025, "date": "2026-01-25"},
            date_bucket=date(2026, 1, 25),
        )
        key2 = compute_idempotency_key(
            source_id="api_football",
            endpoint="fixtures",
            params={"date": "2026-01-25", "league": 140, "season": 2025},  # Reordered
            date_bucket=date(2026, 1, 25),
        )
        assert key1 == key2

    def test_key_length_always_32(self):
        """Key is always exactly 32 characters."""
        # Short params
        key1 = compute_idempotency_key(
            source_id="a",
            endpoint="b",
            params={"x": 1},
            date_bucket=date(2026, 1, 1),
        )
        assert len(key1) == 32

        # Long params
        key2 = compute_idempotency_key(
            source_id="api_football_very_long_source_name",
            endpoint="fixtures/statistics/detailed",
            params={"fixture": 12345, "team": 678, "player": 999, "extra": "value"},
            date_bucket=date(2026, 12, 31),
        )
        assert len(key2) == 32


class TestParamsHash:
    """Test params hash computation."""

    def test_hash_is_md5_length(self):
        """Params hash is MD5 (32 chars)."""
        hash1 = compute_params_hash({"a": 1, "b": 2})
        assert len(hash1) == 32

    def test_hash_is_deterministic(self):
        """Same params always produce same hash."""
        hash1 = compute_params_hash({"fixture": 12345, "bookmaker": 8})
        hash2 = compute_params_hash({"fixture": 12345, "bookmaker": 8})
        assert hash1 == hash2

    def test_hash_order_independent(self):
        """Param order doesn't affect hash."""
        hash1 = compute_params_hash({"a": 1, "b": 2, "c": 3})
        hash2 = compute_params_hash({"c": 3, "a": 1, "b": 2})
        assert hash1 == hash2


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_is_success_for_200(self):
        """HTTP 200 with no error is success."""
        result = ExtractionResult(
            source_id="api_football",
            job_id=uuid4(),
            url="https://example.com/api",
            endpoint="fixtures",
            params_hash="abc123",
            date_bucket=date(2026, 1, 25),
            response_type="json",
            response_body={"data": []},
            http_status=200,
            response_time_ms=100,
            captured_at=date(2026, 1, 25),
            idempotency_key="key123",
        )
        assert result.is_success is True

    def test_is_success_for_201(self):
        """HTTP 201 is also success."""
        result = ExtractionResult(
            source_id="api_football",
            job_id=uuid4(),
            url="https://example.com/api",
            endpoint="fixtures",
            params_hash="abc123",
            date_bucket=date(2026, 1, 25),
            response_type="json",
            response_body={},
            http_status=201,
            response_time_ms=100,
            captured_at=date(2026, 1, 25),
            idempotency_key="key123",
        )
        assert result.is_success is True

    def test_is_not_success_for_429(self):
        """HTTP 429 is not success."""
        result = ExtractionResult(
            source_id="api_football",
            job_id=uuid4(),
            url="https://example.com/api",
            endpoint="fixtures",
            params_hash="abc123",
            date_bucket=date(2026, 1, 25),
            response_type="error",
            response_body=None,
            http_status=429,
            response_time_ms=100,
            captured_at=date(2026, 1, 25),
            idempotency_key="key123",
            error_type="rate_limit",
        )
        assert result.is_success is False

    def test_is_not_success_for_500(self):
        """HTTP 500 is not success."""
        result = ExtractionResult(
            source_id="api_football",
            job_id=uuid4(),
            url="https://example.com/api",
            endpoint="fixtures",
            params_hash="abc123",
            date_bucket=date(2026, 1, 25),
            response_type="error",
            response_body=None,
            http_status=500,
            response_time_ms=100,
            captured_at=date(2026, 1, 25),
            idempotency_key="key123",
            error_type="http_error",
        )
        assert result.is_success is False

    def test_is_not_success_with_error_type(self):
        """Any error_type means not success even with 200."""
        result = ExtractionResult(
            source_id="api_football",
            job_id=uuid4(),
            url="https://example.com/api",
            endpoint="fixtures",
            params_hash="abc123",
            date_bucket=date(2026, 1, 25),
            response_type="error",
            response_body={"errors": ["Invalid token"]},
            http_status=200,  # API returned 200 but with error in body
            response_time_ms=100,
            captured_at=date(2026, 1, 25),
            idempotency_key="key123",
            error_type="api_error",
        )
        assert result.is_success is False

    def test_to_dict_contains_all_fields(self):
        """to_dict() includes all fields for DB insertion."""
        job_id = uuid4()
        captured = date(2026, 1, 25)

        result = ExtractionResult(
            source_id="api_football",
            job_id=job_id,
            url="https://example.com/api",
            endpoint="fixtures",
            params_hash="abc123def456",
            date_bucket=date(2026, 1, 25),
            response_type="json",
            response_body={"data": [1, 2, 3]},
            http_status=200,
            response_time_ms=150,
            captured_at=captured,
            idempotency_key="key123456789012345678901234567890",
        )

        d = result.to_dict()
        assert d["source_id"] == "api_football"
        assert d["job_id"] == job_id
        assert d["url"] == "https://example.com/api"
        assert d["endpoint"] == "fixtures"
        assert d["params_hash"] == "abc123def456"
        assert d["date_bucket"] == date(2026, 1, 25)
        assert d["response_type"] == "json"
        assert d["response_body"] == {"data": [1, 2, 3]}
        assert d["http_status"] == 200
        assert d["response_time_ms"] == 150
        assert d["captured_at"] == captured
        assert d["idempotency_key"] == "key123456789012345678901234567890"
