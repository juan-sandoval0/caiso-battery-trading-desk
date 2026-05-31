"""
MarketIntelAgent — LLM-powered qualitative market intelligence agent.

Uses Claude (via Anthropic API) to parse and summarize:
    - CAISO market notices (operational notices, outage alerts)
    - CAISO Emergency Operating Procedures (EOP) activations
    - Renewable curtailment reports
    - NWS / NOAA weather alerts for California

The agent produces a structured summary that other agents (especially RiskMonitor)
consume to apply qualitative context on top of quantitative signals.

Data sources (scraped, no auth required):
    - CAISO market notices: http://www.caiso.com/market/Pages/MarketNotices/Default.aspx
    - CAISO outage reports: gridstatus get_curtailed_non_operational_generator_report()
    - NWS API: https://api.weather.gov/alerts/active?area=CA (JSON, no key)

Claude model: anthropic/claude-sonnet-4.6 (fast, sufficient for text summarization),
served via OpenRouter using the Anthropic-compatible Messages API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx


# Claude model for market intelligence summarization.
# OpenRouter slug for the same Anthropic model (claude-sonnet-4-6).
_CLAUDE_MODEL: str = "anthropic/claude-sonnet-4.6"

# OpenRouter base URL. The anthropic SDK appends "/v1/messages", so this must
# NOT include the trailing "/v1" (that would double it).
_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api"

# CAISO market notices page (HTML scrape)
_CAISO_NOTICES_URL: str = (
    "http://www.caiso.com/market/Pages/MarketNotices/Default.aspx"
)

# NWS active alerts API for California
_NWS_ALERTS_URL: str = "https://api.weather.gov/alerts/active?area=CA"

# System prompt for Claude
_SYSTEM_PROMPT: str = """You are a CAISO electricity market analyst assistant.
Given raw text from CAISO market notices, outage reports, and weather alerts,
extract and return structured JSON with the following fields:
{
  "eea_level": null | 1 | 2 | 3,
  "has_curtailment_event": bool,
  "curtailment_mw_estimated": float | null,
  "has_major_outage": bool,
  "outage_summary": str | null,
  "weather_risk": "none" | "heat_wave" | "cold_snap" | "wind_event",
  "dispatch_recommendation": str,
  "key_events": [{"type": str, "description": str, "severity": "low"|"medium"|"high"}],
  "raw_summary": str
}
Be concise. Only include events relevant to battery storage dispatch decisions."""


@dataclass
class MarketIntelSummary:
    """Structured output from the MarketIntelAgent.

    Attributes:
        eea_level: CAISO Energy Emergency Alert level (None if no alert active).
        has_curtailment_event: True if significant curtailment is ongoing.
        curtailment_mw_estimated: Estimated curtailment volume in MW.
        has_major_outage: True if a major generation or transmission outage is active.
        outage_summary: Brief text description of the outage.
        weather_risk: Category of weather-driven demand risk.
        dispatch_recommendation: Plain-text guidance for the optimizer.
        key_events: List of structured event dicts.
        raw_summary: Full Claude-generated narrative summary.
        sources: List of URLs or source names that produced this summary.
    """

    eea_level: int | None = None
    has_curtailment_event: bool = False
    curtailment_mw_estimated: float | None = None
    has_major_outage: bool = False
    outage_summary: str | None = None
    weather_risk: str = "none"
    dispatch_recommendation: str = ""
    key_events: list[dict[str, str]] = field(default_factory=list)
    raw_summary: str = ""
    sources: list[str] = field(default_factory=list)


class MarketIntelAgent:
    """LLM-powered qualitative market intelligence parser.

    This agent scrapes raw text from CAISO and NWS, passes it to Claude,
    and returns a structured MarketIntelSummary for other agents to consume.

    Usage::

        agent = MarketIntelAgent(api_key="sk-or-v1-...")
        summary = agent.run()
        # Check summary.eea_level, summary.has_curtailment_event, etc.
    """

    def __init__(
        self,
        api_key: str,
        caiso_fetcher: Any | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
    ) -> None:
        """Initialize the agent with OpenRouter credentials.

        Args:
            api_key: OpenRouter API key (sk-or-...). Routed through the
                Anthropic-compatible Messages API to the same Claude model.
            caiso_fetcher: Optional CAISOFetcher for structured curtailment data.
            base_url: OpenRouter base URL (the anthropic SDK appends /v1/messages).
        """
        import anthropic
        self._base_url = base_url
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self._caiso_fetcher = caiso_fetcher
        self._http = httpx.Client(timeout=30)

    def run(self) -> MarketIntelSummary:
        """Fetch market data, invoke Claude, and return structured summary.

        Returns:
            MarketIntelSummary with parsed market intelligence.
        """
        sources: list[str] = []

        caiso_text = self._fetch_caiso_notices()
        if caiso_text:
            sources.append(_CAISO_NOTICES_URL)

        nws_text = self._fetch_nws_alerts()
        if nws_text:
            sources.append(_NWS_ALERTS_URL)

        combined = "\n\n".join(filter(None, [caiso_text, nws_text]))
        if not combined.strip():
            # No data fetched — return a safe default
            return MarketIntelSummary(
                raw_summary="No market intelligence data available.",
                sources=sources,
            )

        raw_response = self._call_claude(combined)
        summary = self._parse_claude_response(raw_response)
        summary.sources = sources
        return summary

    async def arun(self) -> MarketIntelSummary:
        """Async version of run() for use in the LangGraph async graph.

        Returns:
            MarketIntelSummary.
        """
        import anthropic as _anthropic

        sources: list[str] = []
        caiso_text = ""
        nws_text = ""

        async with httpx.AsyncClient(timeout=30) as async_http:
            try:
                resp = await async_http.get(_CAISO_NOTICES_URL, timeout=15)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                caiso_text = soup.get_text(separator=" ", strip=True)[:3000]
                sources.append(_CAISO_NOTICES_URL)
            except Exception:
                pass

            try:
                resp = await async_http.get(_NWS_ALERTS_URL, timeout=10)
                data = resp.json()
                headlines = [
                    f['properties'].get('headline', '')
                    for f in data.get('features', [])
                ]
                nws_text = "\n".join(headlines[:10])
                sources.append(_NWS_ALERTS_URL)
            except Exception:
                pass

        combined = "\n\n".join(filter(None, [caiso_text, nws_text]))
        if not combined.strip():
            return MarketIntelSummary(raw_summary="No market intelligence data available.", sources=sources)

        async_client = _anthropic.AsyncAnthropic(
            api_key=self._client.api_key, base_url=self._base_url
        )
        try:
            msg = await async_client.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": combined}],
            )
            raw_response = msg.content[0].text
        except Exception:
            return MarketIntelSummary(raw_summary=combined[:500], sources=sources)
        finally:
            await async_client.close()

        summary = self._parse_claude_response(raw_response)
        summary.sources = sources
        return summary

    def _fetch_caiso_notices(self) -> str:
        """Scrape CAISO market notices page and return raw text.

        Returns:
            Extracted text content from the notices page.
        """
        try:
            from bs4 import BeautifulSoup
            resp = self._http.get(_CAISO_NOTICES_URL, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text[:3000]
        except Exception:
            return ""

    def _fetch_nws_alerts(self) -> str:
        """Fetch active NWS weather alerts for California.

        Returns:
            JSON string of alert features, truncated for token budget.
        """
        try:
            resp = self._http.get(_NWS_ALERTS_URL, timeout=10)
            data = resp.json()
            headlines = [
                f["properties"].get("headline", "")
                for f in data.get("features", [])
            ]
            return "\n".join(headlines[:10])
        except Exception:
            return ""

    def _call_claude(self, raw_text: str) -> str:
        """Send raw market text to Claude and return the structured JSON response.

        Args:
            raw_text: Combined text from CAISO notices + NWS alerts.

        Returns:
            Raw JSON string from Claude's response.
        """
        msg = self._client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_text}],
        )
        return msg.content[0].text

    def _parse_claude_response(self, json_str: str) -> MarketIntelSummary:
        """Parse Claude's JSON output into a MarketIntelSummary dataclass.

        Args:
            json_str: Raw JSON string from Claude.

        Returns:
            MarketIntelSummary dataclass.
        """
        # Strip markdown code fences if Claude wrapped the JSON
        clean = re.sub(r"^```(?:json)?\n?|```$", "", json_str.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return MarketIntelSummary(raw_summary=json_str)

        return MarketIntelSummary(
            eea_level=data.get("eea_level"),
            has_curtailment_event=bool(data.get("has_curtailment_event", False)),
            curtailment_mw_estimated=data.get("curtailment_mw_estimated"),
            has_major_outage=bool(data.get("has_major_outage", False)),
            outage_summary=data.get("outage_summary"),
            weather_risk=data.get("weather_risk", "none"),
            dispatch_recommendation=data.get("dispatch_recommendation", ""),
            key_events=data.get("key_events", []),
            raw_summary=data.get("raw_summary", json_str),
        )

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for the orchestrator.

        Returns:
            List of Tool objects: 'fetch_market_intel', 'get_eea_status'.
        """
        from langchain_core.tools import tool

        agent_ref = self

        @tool
        def fetch_market_intel() -> str:
            """Fetch and summarize current CAISO market intelligence. Returns JSON."""
            summary = agent_ref.run()
            return json.dumps({
                "eea_level": summary.eea_level,
                "has_curtailment_event": summary.has_curtailment_event,
                "weather_risk": summary.weather_risk,
                "dispatch_recommendation": summary.dispatch_recommendation,
                "raw_summary": summary.raw_summary[:500],
            })

        @tool
        def get_eea_status() -> str:
            """Get current CAISO Energy Emergency Alert level. Returns JSON."""
            summary = agent_ref.run()
            return json.dumps({
                "eea_level": summary.eea_level,
                "is_emergency": summary.eea_level is not None,
                "dispatch_recommendation": summary.dispatch_recommendation,
            })

        return [fetch_market_intel, get_eea_status]

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()
