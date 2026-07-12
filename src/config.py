"""Application configuration.

Uses dataclasses to keep configuration structured, typed, and immutable
at runtime.  Values can be overridden via environment variables or by
constructing the dataclass with explicit arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.constants import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_SIGNIFICANCE_LEVEL,
    OUTPUT_DIR,
)

@dataclass(frozen=True)
class StatisticsConfig:
    """Parameters governing statistical analysis."""

    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL


@dataclass(frozen=True)
class LLMConfig:
    """Parameters for the LLM integration."""

    model_name: str = "gemini-3.1-flash-lite"
    temperature: float = 0.3
    max_tokens: int = 4096
    api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", ""),
        repr=False,
    )


@dataclass(frozen=True)
class GoogleSheetsConfig:
    """Parameters for the Google Sheets integration."""

    spreadsheet_id: str = "1wFTTRbZE7n3n14-grqkKOvcixNGfJi9Rd1BVxR8XKZc"
    credentials_path: Path = Path("credentials.json")


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    Aggregates all sub-configurations into a single object that can be
    passed through the pipeline.
    """

    output_dir: Path = OUTPUT_DIR
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    google_sheets: GoogleSheetsConfig = field(default_factory=GoogleSheetsConfig)