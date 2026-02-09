"""
Provider abstraction layer for music generation.
Supports multiple backends: YuE (local) and Suno (API).
"""

from enum import Enum
from typing import Literal


class ProviderType(str, Enum):
    YUE = "yue"
    SUNO = "suno"


# Type alias for provider literals
ProviderLiteral = Literal["yue", "suno"]
