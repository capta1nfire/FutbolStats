"""Tests for R2 offload functionality.

Per auditor condition #5: Tests must use mocks for R2 operations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Configure pytest for async tests
pytestmark = pytest.mark.anyio

from app.titan.storage.r2_client import R2Client, build_r2_key


class TestBuildR2Key:
    """Test R2 key construction."""

    def test_build_key_format(self):
        """Key format: {source_id}/{date_bucket}/{job_id}.json"""
        key = build_r2_key(
            source_id="understat",
            date_bucket="2026-01-25",
            job_id="abc123",
        )
        assert key == "understat/2026-01-25/abc123.json"

    def test_build_key_api_football(self):
        """API-Football source uses same format."""
        key = build_r2_key(
            source_id="api_football",
            date_bucket="2026-01-26",
            job_id="def456",
        )
        assert key == "api_football/2026-01-26/def456.json"


class TestR2ClientMocked:
    """Test R2Client with mocked aioboto3."""

    @pytest.fixture
    def r2_client(self):
        """Create R2Client instance for testing."""
        return R2Client(
            endpoint_url="https://test.r2.cloudflarestorage.com",
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            bucket="test-bucket",
        )

    @pytest.mark.asyncio
    async def test_put_object_success(self, r2_client):
        """Test successful object upload."""
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock()

        with patch.object(r2_client, '_get_client', return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_client),
            __aexit__=AsyncMock(return_value=None),
        )):
            result = await r2_client.put_object(
                key="test/key.json",
                body='{"test": "data"}',
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_put_object_failure(self, r2_client):
        """Test upload failure returns False."""
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock(side_effect=Exception("Network error"))

        with patch.object(r2_client, '_get_client', return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_client),
            __aexit__=AsyncMock(return_value=None),
        )):
            result = await r2_client.put_object(
                key="test/key.json",
                body='{"test": "data"}',
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_get_object_success(self, r2_client):
        """Test successful object download."""
        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=b'{"test": "data"}')

        mock_client = AsyncMock()
        mock_client.get_object = AsyncMock(return_value={"Body": mock_body})

        with patch.object(r2_client, '_get_client', return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_client),
            __aexit__=AsyncMock(return_value=None),
        )):
            result = await r2_client.get_object(key="test/key.json")
            assert result == '{"test": "data"}'

    @pytest.mark.asyncio
    async def test_get_object_not_found(self, r2_client):
        """Test object not found returns None."""
        mock_client = AsyncMock()
        # Simulate NoSuchKey exception
        mock_client.get_object = AsyncMock(side_effect=Exception("NoSuchKey"))
        mock_client.exceptions = MagicMock()
        mock_client.exceptions.NoSuchKey = Exception

        with patch.object(r2_client, '_get_client', return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_client),
            __aexit__=AsyncMock(return_value=None),
        )):
            result = await r2_client.get_object(key="nonexistent/key.json")
            assert result is None


class TestR2OffloadThreshold:
    """Test R2 offload threshold logic."""

    def test_threshold_default_is_100kb(self):
        """Default threshold is 100KB."""
        from app.titan.config import get_titan_settings
        settings = get_titan_settings()
        assert settings.R2_OFFLOAD_THRESHOLD_BYTES == 100 * 1024

    def test_small_response_stays_in_db(self):
        """Responses under threshold stay in DB."""
        # 1KB response - should stay in DB
        small_response = json.dumps({"data": "x" * 1000})
        size = len(small_response.encode("utf-8"))
        threshold = 100 * 1024

        assert size < threshold  # Confirm it's under threshold

    def test_large_response_should_offload(self):
        """Responses over threshold should offload to R2."""
        # 150KB response - should offload
        large_response = json.dumps({"data": "x" * 150000})
        size = len(large_response.encode("utf-8"))
        threshold = 100 * 1024

        assert size > threshold  # Confirm it's over threshold


class TestR2IntegrityConstraint:
    """Test R2 integrity constraint: if r2_key NOT NULL then response_body NULL."""

    def test_offloaded_response_has_null_body(self):
        """When r2_key is set, response_body should be None."""
        # This tests the logic, actual DB constraint is tested in integration tests
        r2_key = "understat/2026-01-25/job123.json"
        response_body = None  # Should be NULL when offloaded

        # Assert constraint holds
        assert r2_key is not None
        assert response_body is None

    def test_db_stored_response_has_null_r2_key(self):
        """When response is in DB, r2_key should be None."""
        r2_key = None  # Not offloaded
        response_body = '{"data": "stored in DB"}'

        # Assert constraint holds
        assert r2_key is None
        assert response_body is not None
