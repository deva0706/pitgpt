"""
theme.py — the F1 pit-wall visual identity for the Gradio app.

Kept separate from the app logic so the look can be tweaked without touching
behaviour. It defines a dark, high-contrast "pit wall at night" palette with the
classic F1 red accent, plus a header banner.
"""

from __future__ import annotations

# Injected CSS. Scoped to a few classes we attach in gradio_app.py.
CUSTOM_CSS = """
:root {
    --f1-red: #e10600;
    --f1-dark: #15151e;
    --f1-panel: #1f1f2b;
    --f1-text: #f4f4f6;
}
.gradio-container {
    background: var(--f1-dark) !important;
    color: var(--f1-text) !important;
}
#pitgpt-header {
    background: linear-gradient(90deg, var(--f1-red) 0%, #8b0400 100%);
    color: #fff;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 18px rgba(225, 6, 0, 0.25);
}
#pitgpt-header h1 { margin: 0; font-weight: 800; letter-spacing: 1px; }
#pitgpt-header p { margin: 4px 0 0; opacity: 0.9; font-size: 0.95rem; }
.pitgpt-status {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(255,255,255,0.15);
}
footer { visibility: hidden; }
"""

HEADER_HTML = """
<div id="pitgpt-header">
    <h1>🏎️ PitGPT — Race Engineer</h1>
    <p>Offline F1 strategist on {model} + QLoRA · <span class="pitgpt-status">{status}</span></p>
</div>
"""

# Example prompts shown as clickable chips to get the user started.
EXAMPLE_PROMPTS = [
    "Should we undercut the car ahead or stay out?",
    "Explain the trade-offs between soft, medium and hard tyres.",
    "How does DRS work and when can I use it?",
    "It's starting to rain — inters or stay on slicks?",
    "Give me a radio call to push for the next five laps.",
]
