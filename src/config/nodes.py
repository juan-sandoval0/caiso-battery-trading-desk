"""
CAISO pricing node and trading hub configuration.
"""

from __future__ import annotations

CAISO_HUB_NODES: list[str] = [
    "TH_NP15_GEN-APND",
    "TH_SP15_GEN-APND",
    "TH_ZP26_GEN-APND",
]

DEFAULT_HUB_NODE: str = "TH_NP15_GEN-APND"

MARKET_DA_HOURLY: str = "DAY_AHEAD_HOURLY"
MARKET_RT_5MIN: str = "REAL_TIME_5_MIN"
MARKET_RT_15MIN: str = "REAL_TIME_15_MIN"

CAISO_TZ: str = "US/Pacific"


def node_short_name(node: str) -> str:
    """Return a short display label for a hub node identifier.

    Args:
        node: Full CAISO PNode string (e.g., 'TH_NP15_GEN-APND').

    Returns:
        Short label (e.g., 'NP15').

    >>> node_short_name('TH_NP15_GEN-APND')
    'NP15'
    """
    return node.split("_")[1]
