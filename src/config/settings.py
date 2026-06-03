"""
Environment variable loader and global logging configuration.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed wrapper around environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MarketIntelAgent routes the same Anthropic model through OpenRouter.
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api", alias="OPENROUTER_BASE_URL"
    )
    gridstatus_api_key: str = Field(default="", alias="GRIDSTATUS_API_KEY")
    duckdb_path: Path = Field(default=Path("data/market.duckdb"), alias="DUCKDB_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    default_node: str = Field(default="TH_NP15_GEN-APND", alias="DEFAULT_NODE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    """Initialize structlog with JSON output and silence noisy third-party loggers.

    Args:
        level: Python logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Silence gridstatus and httpx unless user explicitly wants DEBUG.
    # gridstatus uses sub-loggers (gridstatus.caiso, etc.); setting the parent
    # suppresses all of them. Also set root logger level explicitly.
    if numeric_level > logging.DEBUG:
        logging.getLogger().setLevel(numeric_level)
        for noisy in ("gridstatus", "gridstatus.caiso", "httpx", "httpcore", "urllib3", "root"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if level.upper() == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
