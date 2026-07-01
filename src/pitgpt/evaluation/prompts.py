"""
prompts.py — the canonical evaluation prompt set.

These 20 F1 race-engineering questions are the fixed yardstick used in BOTH
Phase 1 (baseline, base model only) and Phase 5 (base vs fine-tuned). Keeping
them in one module guarantees we always compare like-for-like.

The prompts deliberately span the syllabus the persona is meant to master:
strategy, tyres, pit windows, qualifying, telemetry, regulations, DRS, ERS,
weather, and technical driver comparison.
"""

EVAL_PROMPTS: list[str] = [
    # --- Strategy ---
    "Explain the difference between a one-stop and a two-stop race strategy.",
    "When would you call your driver in for an undercut versus an overcut?",
    "How does a safety car change your pit strategy?",
    "What is a 'pit window' and how do you calculate it?",
    # --- Tyres ---
    "Explain the trade-offs between the soft, medium, and hard tyre compounds.",
    "What is thermal degradation and how does it differ from wear?",
    "How do you manage tyre graining during a stint?",
    "Why might a team choose to start on the medium tyre instead of the soft?",
    # --- Qualifying ---
    "Walk me through how you would prepare a car for a qualifying lap.",
    "What is a tow in qualifying and when is it useful?",
    # --- Telemetry & car ---
    "What telemetry channels do you watch to detect brake overheating?",
    "How do you read a throttle trace to judge corner exit performance?",
    "Explain what understeer and oversteer look like in the data.",
    # --- Regulations ---
    "How does the current F1 budget cap affect car development?",
    "Explain the parc fermé rules and what they restrict.",
    # --- DRS / ERS ---
    "How does DRS work and when can a driver use it?",
    "Explain how the ERS deploys energy over a lap.",
    # --- Weather ---
    "How do you decide when to switch from intermediates to slicks on a drying track?",
    # --- Driver comparison ---
    "Compare a driver who is strong on tyre management with one who is strong on qualifying pace.",
    # --- Radio-style ---
    "Give me a radio message telling my driver to push now because the car behind just pitted.",
]

assert len(EVAL_PROMPTS) == 20, "Evaluation set must contain exactly 20 prompts."
