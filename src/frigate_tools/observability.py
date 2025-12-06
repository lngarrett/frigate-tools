"""Observability infrastructure with OpenTelemetry and structlog.

Provides tracing, metrics, and structured logging with trace context injection.
Configure via environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
    OTEL_SERVICE_NAME: Service name (default: frigate-tools)
    OTEL_ENABLED: Enable/disable OTel (default: true)
"""

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Generator, ParamSpec, TypeVar

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode, Tracer

P = ParamSpec("P")
R = TypeVar("R")

_initialized = False
_tracer: Tracer | None = None
_logger: structlog.stdlib.BoundLogger | None = None


def _add_trace_context(
    logger: structlog.stdlib.BoundLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor that adds trace context to log entries."""
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def _is_otel_enabled() -> bool:
    """Check if OTel is enabled via environment variable."""
    return os.environ.get("OTEL_ENABLED", "true").lower() in ("true", "1", "yes")


def _get_endpoint() -> str:
    """Get OTLP endpoint from environment."""
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def _get_service_name() -> str:
    """Get service name from environment."""
    return os.environ.get("OTEL_SERVICE_NAME", "frigate-tools")


def init_observability(
    service_name: str | None = None,
    endpoint: str | None = None,
    sync_export: bool = False,
) -> None:
    """Initialize OpenTelemetry tracing and structlog.

    Args:
        service_name: Override service name (default from OTEL_SERVICE_NAME env var)
        endpoint: Override OTLP endpoint (default from OTEL_EXPORTER_OTLP_ENDPOINT env var)
        sync_export: Use synchronous span export (useful for testing)
    """
    global _initialized, _tracer, _logger

    if _initialized:
        return

    service_name = service_name or _get_service_name()
    endpoint = endpoint or _get_endpoint()

    # Configure OpenTelemetry tracer
    if _is_otel_enabled():
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = (
            SimpleSpanProcessor(exporter) if sync_export else BatchSpanProcessor(exporter)
        )
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)

    # Configure structlog with trace context injection
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_trace_context,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _logger = structlog.get_logger()
    _initialized = True


def get_tracer() -> Tracer:
    """Get the configured tracer, initializing if needed."""
    if not _initialized:
        init_observability()
    assert _tracer is not None
    return _tracer


def get_logger() -> structlog.stdlib.BoundLogger:
    """Get the configured logger, initializing if needed."""
    if not _initialized:
        init_observability()
    assert _logger is not None
    return _logger


@contextmanager
def traced_operation(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager for creating a traced span with logging.

    Args:
        name: Name of the operation/span
        attributes: Optional attributes to add to the span

    Yields:
        The active span

    Example:
        with traced_operation("process_video", {"camera": "front"}) as span:
            # work happens here
            span.set_attribute("frames_processed", 1000)
    """
    tracer = get_tracer()
    logger = get_logger()

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        logger.info(f"Starting {name}", operation=name, **(attributes or {}))

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Completed {name}", operation=name)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error(f"Failed {name}", operation=name, error=str(e))
            raise


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing function calls.

    Args:
        name: Span name (defaults to function name)
        attributes: Optional attributes to add to the span

    Example:
        @traced("encode_video", {"codec": "h264"})
        def encode_video(input_path: str, output_path: str) -> None:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with traced_operation(span_name, attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def shutdown_observability() -> None:
    """Shutdown the tracer provider, flushing any pending spans."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
