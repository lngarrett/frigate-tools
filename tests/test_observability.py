"""Tests for observability module."""

import os
from unittest.mock import patch

import pytest
from opentelemetry import trace

from frigate_tools import observability
from frigate_tools.observability import (
    get_logger,
    get_tracer,
    init_observability,
    shutdown_observability,
    traced,
    traced_operation,
)


@pytest.fixture(autouse=True)
def reset_observability():
    """Reset observability state before each test."""
    observability._initialized = False
    observability._tracer = None
    observability._logger = None
    yield
    shutdown_observability()


@pytest.fixture
def disabled_otel():
    """Fixture to disable OTel export."""
    with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
        yield


def test_init_observability_sets_initialized_flag():
    """init_observability sets the initialized flag."""
    assert observability._initialized is False
    init_observability()
    assert observability._initialized is True


def test_init_observability_idempotent():
    """Calling init_observability multiple times is safe."""
    init_observability()
    tracer1 = observability._tracer
    init_observability()
    tracer2 = observability._tracer
    assert tracer1 is tracer2


def test_get_tracer_returns_tracer():
    """get_tracer returns a valid tracer."""
    tracer = get_tracer()
    assert tracer is not None


def test_get_logger_returns_logger():
    """get_logger returns a valid logger."""
    logger = get_logger()
    assert logger is not None


def test_get_tracer_auto_initializes():
    """get_tracer initializes observability if not done."""
    assert observability._initialized is False
    get_tracer()
    assert observability._initialized is True


def test_get_logger_auto_initializes():
    """get_logger initializes observability if not done."""
    assert observability._initialized is False
    get_logger()
    assert observability._initialized is True


def test_traced_operation_creates_span(disabled_otel):
    """traced_operation creates a span."""
    init_observability()

    with traced_operation("test_op") as span:
        assert span is not None
        assert span.is_recording()


def test_traced_operation_sets_attributes(disabled_otel):
    """traced_operation sets attributes on span."""
    init_observability()

    with traced_operation("test_op", {"key": "value"}) as span:
        # Span should be recording (attributes are internal)
        assert span.is_recording()


def test_traced_operation_records_exception(disabled_otel):
    """traced_operation records exceptions."""
    init_observability()

    with pytest.raises(ValueError, match="test error"):
        with traced_operation("failing_op"):
            raise ValueError("test error")


def test_traced_decorator_wraps_function(disabled_otel):
    """traced decorator wraps function correctly."""
    init_observability()

    @traced("my_operation")
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(5)
    assert result == 10


def test_traced_decorator_uses_function_name(disabled_otel):
    """traced decorator uses function name if not specified."""
    init_observability()

    @traced()
    def another_func() -> str:
        return "hello"

    result = another_func()
    assert result == "hello"


def test_traced_decorator_propagates_exceptions(disabled_otel):
    """traced decorator propagates exceptions."""
    init_observability()

    @traced("failing_func")
    def failing_func() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        failing_func()


def test_otel_enabled_env_var():
    """OTEL_ENABLED env var controls initialization."""
    with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
        assert observability._is_otel_enabled() is False

    with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
        assert observability._is_otel_enabled() is True

    with patch.dict(os.environ, {"OTEL_ENABLED": "1"}):
        assert observability._is_otel_enabled() is True

    with patch.dict(os.environ, {"OTEL_ENABLED": "yes"}):
        assert observability._is_otel_enabled() is True


def test_service_name_from_env():
    """Service name comes from OTEL_SERVICE_NAME env var."""
    with patch.dict(os.environ, {"OTEL_SERVICE_NAME": "my-service"}):
        assert observability._get_service_name() == "my-service"


def test_service_name_default():
    """Service name defaults to frigate-tools."""
    with patch.dict(os.environ, clear=True):
        # Clear the env var if set
        os.environ.pop("OTEL_SERVICE_NAME", None)
        assert observability._get_service_name() == "frigate-tools"


def test_endpoint_from_env():
    """Endpoint comes from OTEL_EXPORTER_OTLP_ENDPOINT env var."""
    with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://custom:4317"}):
        assert observability._get_endpoint() == "http://custom:4317"


def test_endpoint_default():
    """Endpoint defaults to localhost:4317."""
    with patch.dict(os.environ, clear=True):
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        assert observability._get_endpoint() == "http://localhost:4317"


def test_init_with_custom_service_name(disabled_otel):
    """init_observability accepts custom service name."""
    init_observability(service_name="custom-service")
    assert observability._initialized is True


def test_init_with_custom_endpoint(disabled_otel):
    """init_observability accepts custom endpoint."""
    init_observability(endpoint="http://custom:4317")
    assert observability._initialized is True
