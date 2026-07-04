"""
theme.py — the visual identity for the PitGPT chat app.

Aesthetic direction: **"pit wall broadcast"** (spec:
docs/superpowers/specs/2026-07-04-pitwall-ui-design.md). Dark carbon surfaces,
timing-tower typography, sharp edges, F1-red and telemetry-teal accents.
Messages render as a race-radio transcript — labeled rows with edge bars —
not floaty chat bubbles.

Fully offline by design: system fonts only (Segoe UI + Cascadia Mono
fallbacks), no CDN imports.

Kept separate from app logic so the look can evolve without touching
behaviour. ``build_theme()`` sets the baseline and ``CUSTOM_CSS`` layers the
header, transcript rows, and interactions.
"""

from __future__ import annotations

import gradio as gr

_SANS = "'Segoe UI', 'Inter', system-ui, sans-serif"
_MONO = "'Cascadia Mono', 'Consolas', ui-monospace, monospace"


def build_theme() -> gr.themes.Base:
    """Baseline Gradio theme: carbon palette + system fonts."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        # LocalFont keeps the app fully offline (no Google Fonts CDN), and
        # Gradio 6 needs Font OBJECTS here — plain strings crash the theme
        # comparison ('str' object has no attribute 'name').
        font=[gr.themes.LocalFont("Segoe UI"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.LocalFont("Cascadia Mono"), "Consolas", "monospace"],
    ).set(
        body_background_fill="#0e0f12",
        body_text_color="#f2f3f5",
        body_text_color_subdued="#9aa1ab",
        block_background_fill="#16181d",
        block_border_color="#262a31",
        block_label_text_color="#9aa1ab",
        input_background_fill="#101216",
        input_border_color="#2c313a",
        button_primary_background_fill="#e10600",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="transparent",
        button_secondary_text_color="#9aa1ab",
    )


# Injected CSS. Loaded via demo.launch(css=CUSTOM_CSS).
CUSTOM_CSS = f"""
:root {{
    --pw-bg:     #0e0f12;
    --pw-panel:  #16181d;
    --pw-line:   #262a31;
    --pw-red:    #e10600;
    --pw-teal:   #00d2be;
    --pw-text:   #f2f3f5;
    --pw-muted:  #9aa1ab;
}}

.gradio-container {{
    background: var(--pw-bg) !important;
    color: var(--pw-text) !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    font-family: {_SANS} !important;
}}
footer {{ display: none !important; }}

/* ---- Header: timing-tower strip ---- */
#pitwall-header {{
    display: flex; align-items: center; gap: 14px;
    margin: 10px 0 4px; padding: 14px 18px;
    background: linear-gradient(180deg, #191c21, #14161a);
    border: 1px solid var(--pw-line);
    border-top: 3px solid var(--pw-red);
    border-radius: 4px;
}}
.pw-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--pw-red); flex: none;
    animation: pw-pulse 1.6s infinite;
}}
.pw-wordmark {{
    font-family: {_SANS}; font-weight: 900; font-style: italic;
    font-size: 1.5rem; letter-spacing: 0.04em; line-height: 1;
    color: #ffffff !important; text-transform: uppercase;
}}
.pw-wordmark b {{ color: var(--pw-red); }}
.pw-role {{
    font-family: {_MONO}; font-size: 0.66rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--pw-muted);
    border-left: 1px solid var(--pw-line); padding-left: 14px;
}}
.pw-chips {{
    margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap;
}}
.pw-chip {{
    font-family: {_MONO}; font-size: 0.62rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--pw-muted);
    padding: 4px 9px; border: 1px solid var(--pw-line); border-radius: 3px;
    background: #101216;
}}
.pw-chip.hot {{ color: var(--pw-teal); border-color: rgba(0,210,190,0.35); }}

/* ---- Transcript: timing rows, not bubbles ---- */
#pg-chat {{
    border: 1px solid var(--pw-line) !important;
    border-radius: 4px !important;
    background: var(--pw-panel) !important;
}}
#pg-chat .message, #pg-chat [class*="bubble"] {{
    border-radius: 3px !important;
    border: none !important;
    font-size: 0.98rem !important;
    line-height: 1.6 !important;
    font-family: {_SANS} !important;
}}
/* PitGPT rows: teal edge bar + mono label */
#pg-chat [data-testid="bot"], #pg-chat .bot, #pg-chat [class*="bot"] {{
    background: #1a1d23 !important;
    border-left: 3px solid var(--pw-teal) !important;
    color: var(--pw-text) !important;
    position: relative;
    padding-top: 26px !important;
}}
#pg-chat [data-testid="bot"]::before {{
    content: "PITGPT ▮ RACE ENGINEER";
    position: absolute; top: 7px; left: 12px;
    font-family: {_MONO}; font-size: 0.6rem; letter-spacing: 0.2em;
    color: var(--pw-teal); opacity: 0.9;
}}
/* User rows: red edge bar + mono label */
#pg-chat [data-testid="user"], #pg-chat .user, #pg-chat [class*="user"] {{
    background: rgba(225,6,0,0.10) !important;
    border-right: 3px solid var(--pw-red) !important;
    color: #ffffff !important;
    position: relative;
    padding-top: 26px !important;
}}
#pg-chat [data-testid="user"]::before {{
    content: "YOU ▸";
    position: absolute; top: 7px; right: 12px;
    font-family: {_MONO}; font-size: 0.6rem; letter-spacing: 0.2em;
    color: var(--pw-red); opacity: 0.95;
}}
/* Hide per-message icon clutter (copy/share buttons) */
#pg-chat .message-buttons, #pg-chat [class*="icon-button"],
#pg-chat button[title*="opy"], #pg-chat [class*="options"] {{
    display: none !important;
}}

/* ---- Input strip ---- */
#pg-input textarea {{
    font-size: 1rem !important;
    font-family: {_SANS} !important;
    border-radius: 3px !important;
    caret-color: var(--pw-red);
}}
#pg-input textarea:focus {{
    border-color: var(--pw-red) !important;
    box-shadow: 0 0 0 2px rgba(225,6,0,0.25) !important;
}}
#pg-send {{
    border-radius: 3px !important;
    font-family: {_MONO} !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    transition: filter .15s;
}}
#pg-send:hover {{ filter: brightness(1.15); }}
#pg-reset {{
    border: 1px solid var(--pw-line) !important;
    border-radius: 3px !important;
    background: transparent !important;
    color: var(--pw-muted) !important;
    font-family: {_MONO} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}}
#pg-reset:hover {{ color: var(--pw-text) !important; border-color: var(--pw-red) !important; }}

/* ---- Suggested radio calls ---- */
#pg-examples .label, #pg-examples label {{
    font-family: {_MONO} !important; font-size: 0.64rem !important;
    letter-spacing: 0.2em !important; text-transform: uppercase !important;
    color: var(--pw-muted) !important;
}}
#pg-examples button, .gradio-container [class*="example"] button {{
    border-radius: 3px !important;
    border: 1px solid var(--pw-line) !important;
    background: #101216 !important;
    color: var(--pw-muted) !important;
    font-family: {_MONO} !important;
    font-size: 0.78rem !important;
    transition: color .15s, border-color .15s;
}}
#pg-examples button:hover {{
    color: var(--pw-text) !important;
    border-color: var(--pw-teal) !important;
}}

/* Scrollbar */
#pg-chat ::-webkit-scrollbar {{ width: 8px; }}
#pg-chat ::-webkit-scrollbar-thumb {{ background: #2c313a; border-radius: 0; }}

@keyframes pw-pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(225,6,0,0.5); }}
    70%  {{ box-shadow: 0 0 0 7px rgba(225,6,0,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(225,6,0,0); }}
}}
"""


def header_html(status: str, model: str) -> str:
    """Timing-tower header strip with live status chips."""
    return f"""
<div id="pitwall-header">
  <span class="pw-dot"></span>
  <span class="pw-wordmark">Pit<b>GPT</b></span>
  <span class="pw-role">Race Engineer</span>
  <span class="pw-chips">
    <span class="pw-chip">{model}</span>
    <span class="pw-chip hot">{status}</span>
    <span class="pw-chip">100% local</span>
  </span>
</div>
"""


# Fan-friendly conversation starters.
EXAMPLE_PROMPTS = [
    "Who won the 2019 Chinese Grand Prix?",
    "Which driver took the title in 2016?",
    "Explain DRS like I'm five",
    "Senna or Prost — who was better?",
    "Why is Monaco so special?",
    "Who won the 2007 Singapore Grand Prix?",
]
