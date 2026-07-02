"""
theme.py — the visual identity for the PitGPT chat app.

Aesthetic direction: **"night race / trackside telemetry."** A warm dark paddock
at night — deep charcoal, a live "on-air" pulse, an ember-orange accent (brake
glow, not the cliché flat F1 red), tyre-compound accent dots, a distinctive
display wordmark, and a subtle carbon grain. Warm and human to match PitGPT's
lifelong-enthusiast persona, not a sterile corporate F1 look.

Kept separate from app logic so the look can evolve without touching behaviour.
``build_theme()`` sets the baseline (fonts, hues, block colours) and
``CUSTOM_CSS`` layers the atmosphere, header, chat bubbles, and interactions.
"""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.themes.Base:
    """Baseline Gradio theme: distinctive fonts + a warm dark palette."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#131010",
        body_text_color="#f2ece4",
        body_text_color_subdued="#a29a90",
        block_background_fill="rgba(255,255,255,0.028)",
        block_border_color="rgba(255,255,255,0.08)",
        block_label_text_color="#c9b8a8",
        input_background_fill="rgba(255,255,255,0.04)",
        input_border_color="rgba(255,255,255,0.10)",
        button_primary_background_fill="linear-gradient(135deg,#ff7a18 0%,#ff3d2e 100%)",
        button_primary_text_color="#1a0f08",
        button_secondary_background_fill="rgba(255,255,255,0.05)",
        button_secondary_text_color="#f2ece4",
    )


# Injected CSS. Loaded via demo.launch(css=CUSTOM_CSS).
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --pg-bg:      #131010;
    --pg-panel:   rgba(255,255,255,0.028);
    --pg-line:    rgba(255,255,255,0.08);
    --pg-text:    #f2ece4;
    --pg-muted:   #a29a90;
    --pg-ember:   #ff5a1f;
    --pg-ember-2: #ff3d2e;
    --pg-gold:    #ffb020;
    --pg-live:    #34d399;
    /* tyre compounds */
    --pg-soft:    #ef4444;
    --pg-medium:  #f5c518;
    --pg-hard:    #e9e4dc;
}

/* ---- Atmosphere: warm night-race glow + fine carbon grain ---- */
.gradio-container {
    background:
        radial-gradient(1100px 500px at 78% -8%, rgba(255,90,31,0.16), transparent 60%),
        radial-gradient(900px 500px at 8% 108%, rgba(255,176,32,0.08), transparent 55%),
        var(--pg-bg) !important;
    color: var(--pg-text) !important;
    max-width: 940px !important;
    margin: 0 auto !important;
}
.gradio-container::before {   /* carbon grain overlay */
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.5;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
}
.gradio-container > * { position: relative; z-index: 1; }
footer { display: none !important; }

/* ---- Header ---- */
#pitgpt-header {
    margin: 6px 0 18px;
    padding: 26px 30px;
    border-radius: 20px;
    border: 1px solid var(--pg-line);
    background:
        linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)),
        #191411;
    box-shadow: 0 24px 60px -30px rgba(255,90,31,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
    position: relative;
    animation: pg-rise 0.7s cubic-bezier(.2,.7,.2,1) both;
}
#pitgpt-header::after {   /* speed streak */
    content: ""; position: absolute; top: 0; right: -10%; width: 55%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,90,31,0.10) 60%, transparent);
    transform: skewX(-18deg); pointer-events: none;
}
.pg-brand { display: flex; align-items: baseline; gap: 12px; }
.pg-wordmark {
    font-family: 'Bricolage Grotesque', 'Hanken Grotesk', sans-serif;
    font-weight: 800; font-size: 2.6rem; line-height: 1; letter-spacing: -0.02em;
    color: #fff4e9 !important;
    -webkit-text-fill-color: #fff4e9 !important;
    text-shadow: 0 0 22px rgba(255,90,31,0.45), 0 2px 10px rgba(0,0,0,0.4);
}
.pg-wordmark b { color: var(--pg-ember); -webkit-text-fill-color: var(--pg-ember); }
.pg-flag { font-size: 1.7rem; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.5)); }
.pg-live {
    margin-left: auto; display: inline-flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--pg-live);
    padding: 5px 11px; border-radius: 999px;
    background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.25);
}
.pg-live .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--pg-live);
    box-shadow: 0 0 0 0 rgba(52,211,153,0.6); animation: pg-pulse 1.8s infinite;
}
.pg-tagline { margin: 12px 0 0; color: var(--pg-muted); font-size: 0.98rem; max-width: 64ch; }
.pg-compounds { display: flex; align-items: center; gap: 8px; margin-top: 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6f675f; }
.pg-compounds b { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.pg-c-soft { background: var(--pg-soft); }
.pg-c-med  { background: var(--pg-medium); }
.pg-c-hard { background: var(--pg-hard); }
.pg-status { color: var(--pg-gold); font-weight: 600; }

/* ---- Chat surface + bubbles (best-effort across Gradio internals) ---- */
#pg-chat { border: 1px solid var(--pg-line) !important; border-radius: 18px !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent) !important; }
#pg-chat .message, #pg-chat .bubble, #pg-chat [class*="bubble"] {
    border-radius: 16px !important; font-size: 1.0rem !important; line-height: 1.6 !important;
    border: 1px solid var(--pg-line) !important;
}
/* assistant bubbles: warm card with an ember edge */
#pg-chat [data-testid="bot"], #pg-chat .bot, #pg-chat [class*="bot"] {
    background: #1b1613 !important; border-left: 3px solid var(--pg-ember) !important;
    color: var(--pg-text) !important;
}
/* user bubbles: ember fill */
#pg-chat [data-testid="user"], #pg-chat .user, #pg-chat [class*="user"] {
    background: linear-gradient(135deg, rgba(255,122,24,0.20), rgba(255,61,46,0.14)) !important;
    border: 1px solid rgba(255,90,31,0.35) !important; color: #fff !important;
}

/* ---- Input row ---- */
#pg-input textarea {
    font-size: 1.02rem !important; border-radius: 14px !important;
    transition: border-color .2s, box-shadow .2s;
}
#pg-input textarea:focus {
    border-color: var(--pg-ember) !important;
    box-shadow: 0 0 0 3px rgba(255,90,31,0.18) !important;
}
#pg-send { border-radius: 14px !important; font-weight: 700 !important; letter-spacing: 0.01em;
    box-shadow: 0 10px 30px -12px rgba(255,90,31,0.7) !important; transition: transform .12s, filter .2s; }
#pg-send:hover { transform: translateY(-1px); filter: brightness(1.06); }
#pg-reset { border-radius: 12px !important; color: var(--pg-muted) !important; }
#pg-reset:hover { color: var(--pg-text) !important; border-color: var(--pg-ember) !important; }

/* Example chips */
.gradio-container .examples, #pg-examples { animation: pg-rise 0.9s .1s cubic-bezier(.2,.7,.2,1) both; }
#pg-examples button, .gradio-container [class*="example"] button {
    border-radius: 999px !important; border: 1px solid var(--pg-line) !important;
    background: rgba(255,255,255,0.03) !important; color: var(--pg-text) !important;
    transition: border-color .2s, background .2s, transform .1s;
}
#pg-examples button:hover {
    border-color: var(--pg-gold) !important; background: rgba(255,176,32,0.08) !important;
    transform: translateY(-1px);
}

/* Scrollbar */
#pg-chat ::-webkit-scrollbar { width: 10px; }
#pg-chat ::-webkit-scrollbar-thumb { background: rgba(255,90,31,0.35); border-radius: 999px; }

@keyframes pg-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}
@keyframes pg-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
"""


def header_html(status: str, model: str) -> str:
    """Build the header banner. ``status``/``model`` are shown as live telemetry."""
    return f"""
<div id="pitgpt-header">
  <div class="pg-brand">
    <span class="pg-flag">🏁</span>
    <span class="pg-wordmark">Pit<b>GPT</b></span>
    <span class="pg-live"><span class="dot"></span> On the radio</span>
  </div>
  <p class="pg-tagline">
    Your lifelong Formula 1 mate — here to talk drivers, strategy, history and the
    tech, and to make the sport click. Ask me anything, no question too basic.
  </p>
  <div class="pg-compounds">
    <b class="pg-c-soft"></b><b class="pg-c-med"></b><b class="pg-c-hard"></b>
    <span>{model}</span> · <span class="pg-status">{status}</span>
  </div>
</div>
"""


# Fan-friendly conversation starters.
EXAMPLE_PROMPTS = [
    "I'm new to F1 — where do I start?",
    "Senna or Prost — who was better?",
    "Explain DRS like I'm five",
    "Why is Monaco so special?",
    "What's the best race you've ever seen?",
    "Undercut vs overcut — what's the difference?",
]
