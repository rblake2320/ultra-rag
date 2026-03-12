"""
Unified LLM client: Ollama (primary) → Claude API (fallback).
Used for KG extraction, summarization, contextual retrieval, and query routing.

Usage:
    from src.llm import LLMClient
    llm = LLMClient()                          # uses config defaults
    llm = LLMClient(model="gemma3:latest")     # override model
    text = llm.complete("Summarise this…")
    data = llm.complete_json("Extract entities from…")
"""
import json
import logging
import os
import re
from typing import Optional

import httpx

from .config import get_config

log = logging.getLogger(__name__)

# Regex to strip markdown code fences from LLM JSON responses
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE
)

# Default models by task profile
_DEFAULT_EXTRACTION_MODEL = "deepseek-r1:32b"   # high-quality reasoning
_DEFAULT_FAST_MODEL        = "gemma3:latest"     # low-latency tasks
_DEFAULT_OLLAMA_URL        = "http://localhost:11434"


class LLMClient:
    """
    Thin wrapper around Ollama's /api/generate endpoint with a Claude fallback.

    Attributes
    ----------
    provider : str
        "ollama" (default) or "claude".  If Ollama is unreachable the client
        automatically falls back to Claude when ANTHROPIC_API_KEY is set.
    model : str
        Model name sent to Ollama (e.g. "deepseek-r1:32b").
    ollama_url : str
        Base URL for the local Ollama server.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        llm_cfg = cfg.get("llm", {})

        self.provider   = provider
        self.ollama_url = llm_cfg.get("ollama_url", _DEFAULT_OLLAMA_URL).rstrip("/")

        # Model resolution priority: constructor arg > config > hardcoded default
        if model:
            self.model = model
        elif "model" in llm_cfg:
            self.model = llm_cfg["model"]
        else:
            # Pick extraction vs fast default based on provider hint in config
            task = llm_cfg.get("default_task", "extraction")
            self.model = (
                _DEFAULT_FAST_MODEL
                if task == "fast"
                else _DEFAULT_EXTRACTION_MODEL
            )

        # Claude fallback: only initialised when needed
        self._claude_client = None

        log.debug("LLMClient ready — provider=%s model=%s", self.provider, self.model)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2_000,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a completion for *prompt*.

        Parameters
        ----------
        prompt     : User prompt text.
        system     : Optional system/instruction prompt.
        max_tokens : Approximate token budget for the response.
        json_mode  : When True, appends a JSON-only instruction to *system*.

        Returns
        -------
        Stripped response string, or "" on failure.
        """
        effective_system = self._build_system(system, json_mode)

        # Primary: Ollama
        try:
            return self._ollama_complete(prompt, effective_system, max_tokens)
        except Exception as exc:
            log.warning("Ollama completion failed (%s); trying Claude fallback.", exc)

        # Fallback: Claude API
        try:
            return self._claude_complete(prompt, effective_system, max_tokens)
        except Exception as exc:
            log.error("Claude fallback also failed: %s", exc)
            return ""

    def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> dict:
        """
        Like :meth:`complete` but parses the response as JSON.

        Strips markdown code fences, then calls json.loads.

        Returns
        -------
        Parsed dict, or {} on any parse/completion failure.
        """
        raw = self.complete(prompt, system=system, json_mode=True)
        if not raw:
            return {}

        text = raw.strip()

        # Strip ```json ... ``` fences if present
        m = _JSON_FENCE_RE.search(text)
        if m:
            text = m.group(1).strip()

        # Some models prepend <think>…</think> blocks (deepseek-r1)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("JSON parse failed (%s); raw snippet: %.200s", exc, text)
            return {}

    def batch_complete(
        self,
        prompts: list,
        **kwargs,
    ) -> list:
        """
        Complete a list of prompts sequentially.

        Ollama does not support true request batching; this loops over each
        prompt individually.  Pass any keyword args accepted by :meth:`complete`.

        Returns
        -------
        list of str — same length as *prompts*, "" for any failed item.
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                results.append(self.complete(prompt, **kwargs))
            except Exception as exc:
                log.warning("batch_complete[%d] failed: %s", i, exc)
                results.append("")
        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_system(system: Optional[str], json_mode: bool) -> Optional[str]:
        """Combine caller-supplied system prompt with JSON-mode instruction."""
        json_instruction = "Respond with valid JSON only. Do not include any prose, markdown fences, or commentary outside the JSON structure."
        if json_mode:
            if system:
                return f"{system}\n\n{json_instruction}"
            return json_instruction
        return system

    def _ollama_complete(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        """POST to Ollama /api/generate (non-streaming)."""
        payload: dict = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        resp = httpx.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=180.0,  # longer for large models
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"response": "...", "done": true, ...}
        text = data.get("response", "")
        return text.strip()

    def _claude_complete(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        """
        Fall back to Anthropic Claude API.

        Requires ANTHROPIC_API_KEY to be set in the environment.
        Uses the `anthropic` Python package (pip install anthropic).
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set — Claude fallback unavailable.")

        # Lazy-import so the package is optional
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Install 'anthropic' package to enable Claude fallback."
            ) from exc

        if self._claude_client is None:
            self._claude_client = anthropic.Anthropic(api_key=api_key)

        cfg = get_config()
        llm_cfg = cfg.get("llm", {})
        claude_model = llm_cfg.get(
            "claude_fallback_model", "claude-3-haiku-20240307"
        )

        kwargs: dict = {
            "model":      claude_model,
            "max_tokens": max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        message = self._claude_client.messages.create(**kwargs)
        text = message.content[0].text if message.content else ""
        return text.strip()
