"""
engine.py — the chat/generation layer that sits on top of a loaded model.

Responsibilities
----------------
* Decide what a user message needs (``_prepare``): a reply that is already
  determined, a grounded answer from retrieved facts, or free chat.
* Turn a list of chat messages into model-ready input ids using the tokenizer's
  built-in chat template (the template is model-specific; Gemma's is used by default).
* Run ``model.generate()`` with our configured decoding parameters, blocking or
  streamed, and check a grounded reply against its facts before it is shown
  (``answers.py``).

Keeping this separate from ``model_loader`` means the UI and the eval harness can
share exactly one definition of "how PitGPT talks".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from threading import Event, Thread
from typing import Iterable, Iterator

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from pitgpt.config import settings
from pitgpt.data import templates as T
from pitgpt.inference import answers
from pitgpt.inference.dialogue import (
    _real_names,
    follow,
    followup,
    references,
    resolve,
    text_of,
)
from pitgpt.inference.templating import render_chat
from pitgpt.rag.records import parse_race

logger = logging.getLogger(__name__)

# A single chat turn: {"role": "system"|"user"|"assistant", "content": str}.
Message = dict[str, str]

# Some UIs (notably Gradio 6's chatbot) store content as a list of parts.
_content_to_text = text_of


def build_messages(
    user_prompt,
    history: list[Message] | None = None,
    context: str = "",
) -> list[Message]:
    """Assemble the full message list, prepending the enthusiast system prompt.

    ``history`` (if given) is the prior turns so multi-turn context is preserved.
    ``context`` (if given) is a block of retrieved facts (RAG) added as a second
    system message so the model can ground its answer without us baking facts into
    the weights. All content is normalised to plain strings so the chat template
    never sees a list (see :func:`dialogue.text_of`).
    """
    messages: list[Message] = [{"role": "system", "content": settings.model.system_prompt}]
    for turn in history or []:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_to_text(turn.get("content"))
        if text:
            messages.append({"role": role, "content": text})
    # Fold retrieved facts into the *user* turn, right next to the question — small
    # models follow context far better here than in a separate system message.
    user_text = _content_to_text(user_prompt)
    if context:
        user_text = f"{context}\n\nUsing the facts above where relevant, answer: {user_text}"
    messages.append({"role": "user", "content": user_text})
    return messages


def _encode(tokenizer, messages: list[Message], device) -> dict:
    """Apply the chat template and return model inputs on ``device``.

    Newer ``transformers`` (5.x) returns a dict-like ``BatchEncoding`` (with
    ``input_ids`` *and* ``attention_mask``) when ``return_dict=True``; older
    versions returned a bare tensor. We normalise to a dict so callers always get
    an attention mask, which silences generation warnings and is correct for
    left-padded batches.
    """
    encoded = render_chat(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,   # append the "assistant:" cue so the model replies
        return_tensors="pt",
        return_dict=True,
    )
    if hasattr(encoded, "to"):
        # BatchEncoding.to() moves every contained tensor to the device at once.
        return encoded.to(device)
    # Fallback for older transformers that return a bare tensor.
    return {"input_ids": encoded.to(device)}


_CHAT_TOKENS = 80


def _generation_kwargs(grounded: bool = True) -> dict:
    """Translate our GenerationConfig dataclass into generate() kwargs.

    Ungrounded persona turns get a shorter budget: the fabricated years and
    stats in free chat live in sentence four onward of the long rambles
    (measured live — "in '17 he won it" about Alonso). Short and warm beats
    long and wrong."""
    g = settings.generation
    kwargs = {
        "max_new_tokens": g.max_new_tokens if grounded else min(g.max_new_tokens, _CHAT_TOKENS),
        "repetition_penalty": g.repetition_penalty,
        "do_sample": g.do_sample,
    }
    if g.do_sample:      # greedy decoding ignores these, and transformers says so every turn
        kwargs.update(temperature=g.temperature, top_p=g.top_p, top_k=g.top_k)
    return kwargs


# --- what the user has told us ------------------------------------------------ #
# The adapter was never trained on chat-recall contexts, so questions about the
# user's own statements are answered from the transcript, not by the model.
_RECALL_CUE = re.compile(
    r"\b(he|she|they|him|her|his|hers|their|it|that (race|season|driver|team|year)|"
    r"my favou?rite|i (said|mentioned|told|asked)|earlier|before|again|same)\b", re.I)
_SELF_CUE = re.compile(r"\b(my favou?rite|i (said|mentioned|told|asked))\b", re.I)
_MY_FAVOURITE = re.compile(
    r"\bmy favou?rite(?: (?:f1 |formula (?:1|one) )?"
    r"(?:driver|team|circuit|track|race|grand prix|gp|constructor)s?)?\b", re.I)

# "What are MY favourite circuits?" asks about the USER, not about F1 — corpus
# docs are pure noise there, and a retrieved Suzuka doc beat the user's own
# words in live testing ("you love Suzuka!" to someone who never said it).
# Enumeration questions about the user answer from chat recall ALONE.
# Deliberately narrow: "What teams did my favourite driver race for?" still
# needs retrieval and does not match.
_SELF_REFERENTIAL = re.compile(
    # non-anchored: "Tell me what are my favourite GPs" must match too.
    # "What teams did my favourite driver..." stays safe — that's "did my",
    # not "is/are my".
    r"\b(what|which)\s+(is|are|was|were)\s+my\b"
    r"|\btell me my\b"
    r"|\bwhat (did i|have i) (say|said|tell|told|mention(ed)?|ask(ed)?)\b", re.I)

# The radio log must quote what the user STATED, never their questions —
# live it echoed "Tell me what are my favourite Grand Prix" back as a
# "transmission", and a recalled junk question turned an unrelated statement
# turn into a grounded decline about corn coffee.
_QUESTIONISH = re.compile(
    r"\?|^\W*(so\s+|yes\s+|ok(ay)?\s+)*"
    r"(what|who|when|where|which|why|how|tell|explain|give|can|could|do|does|did)\b",
    re.I)

# "...is Fernando Alonso, ok?" is a STATEMENT — the trailing tag-question
# made _QUESTIONISH discard the user's own declaration (measured: the radio
# log replied "nothing on that from you yet" right after it).
_TAG_QUESTION = re.compile(
    r"[,;\s]*\b(ok(ay)?|right|yeah|yes|no|agreed|got it)\?\s*$", re.I)


def _strip_tag_question(text: str) -> str:
    return _TAG_QUESTION.sub("", text).strip()


def _radio_log_reply(user_prompt: str, history: list[Message] | None) -> str | None:
    """Deterministic answer for self-referential questions ("what are my
    favourite circuits?"): quote the user's own logged turns back verbatim.

    The adapter was never trained on chat-recall contexts, and measured live
    it either spotlighted one item or fabricated a fresh list ("Monaco,
    Silverstone, Suzuka" to a user who said Yas Marina and Singapore). Until
    a recall-trained adapter ships, the radio log answers these itself —
    quoting is the one strategy that cannot hallucinate.
    """
    if not (history and _SELF_REFERENTIAL.search(user_prompt)):
        return None
    from pitgpt.rag.retriever import _USER_STATEMENT

    statements = [_strip_tag_question(_content_to_text(t.get("content"))[:220])
                  for t in history[-8:] if t.get("role") == "user"]
    # A plain first-person preference is a transmission too: with only the
    # recall-cue filter, "I also love the Monaco GP." was dropped from "What
    # have I told you so far?" while "My favourite driver is..." was quoted
    # (measured 2026-10-11).
    statements = [s for s in statements
                  if s and (_RECALL_CUE.search(s) or _USER_STATEMENT.search(s))
                  and not _QUESTIONISH.search(s)]
    if not statements:
        return ("Checking the radio log — nothing on that from you yet. "
                "Tell me and I'll keep it on file!")
    quoted = "\n".join(f"▸ “{s}”" for s in statements[-3:])
    return (f"Checking the radio log — here's what you've told me so far:\n"
            f"{quoted}\n"
            f"That's straight from your own transmissions. Want the full "
            f"telemetry on any of them?")


# "Who is my favourite driver?" is a LOOKUP, not an enumeration: the user
# wants their stated name back WITH its data card. Measured 2026-07-07:
# retrieval returned the Alonso bio doc and the adapter still declined — the
# recall-block context shape is untrained (see _radio_log_reply). So the
# recall half is answered deterministically, and the model generates only
# the part it is reliably trained on: the identity card for the named entity.
_FAV_LOOKUP = re.compile(
    r"\bwho(?:'s|\s+(?:is|was))\s+my\s+favou?rite\s*(\w*)", re.I)


def _favourite_lookup(
    user_prompt: str, history: list[Message] | None
) -> tuple[str, str | None] | None:
    """Resolve "who is my favourite {noun}?" from the user's own statements.

    Returns None when the question isn't a favourite lookup. Otherwise
    ``(text, rewritten)``: ``rewritten`` is None for a terminal no-statement
    reply, or a "Tell me about {name}." prompt to generate the data card,
    with ``text`` as its deterministic radio-log prefix. With several stated
    favourites the most recent statement wins.
    """
    m = _FAV_LOOKUP.search(user_prompt)
    if not (history and m):
        return None
    # Scan the WHOLE history, and only the user's own favourite STATEMENTS.
    # A window of recent turns lost the statement after five exchanges, and
    # names from the bot's answers must never qualify — measured live as
    # "your favourite driver on record is Autodromo Nazionale" (a circuit
    # from the previous Monza reply).
    stated: list[str] = []
    for t in history:
        if t.get("role") != "user":
            continue
        text = _strip_tag_question(_content_to_text(t.get("content"))[:220])
        if (not text or _QUESTIONISH.search(text)
                or not re.search(r"favou?rite", text, re.I)):
            continue
        stated.extend(_real_names(text))   # title phrases are not names
    if not stated:
        return ("Checking the radio log — nothing on that from you yet. "
                "Tell me and I'll keep it on file!", None)
    name = stated[-1]
    noun = (m.group(1) or "").lower()
    what = f"favourite {noun}" if noun else "favourite"
    return (f"Checking the radio log — your {what} on record is {name}. ",
            f"Tell me about {name}.")


def _own_words(user_prompt: str, history: list[Message] | None) -> tuple[str, str] | None:
    """Resolve a reference to what the user said earlier ("my favourite
    driver"). None if the question makes none; otherwise ``(block, question)``
    — the question with the stated name written in, or, when no name can be
    read from the statements, those statements labeled for the prompt.

    A preference the user STATED beats names that merely appeared in
    questions or answers ("Who is my favorite driver?" after a Vettel
    statement pulled names from a Hamilton question too, and the mixed
    context produced nonsense); the most recent statement that names anyone
    wins.
    """
    if not (history and _SELF_CUE.search(user_prompt)):
        return None
    # Gradio 6 stores content as a LIST of parts — a plain isinstance(str)
    # filter here silently dropped every real turn (recall worked in tests,
    # never in the app). Flatten like build_messages does.
    said = [_strip_tag_question(_content_to_text(t.get("content"))[:220])
            for t in history[-8:] if t.get("role") == "user"]
    # User QUESTIONS stay out: a recalled junk question in context derailed
    # an unrelated turn live.
    said = [s for s in said if s and _RECALL_CUE.search(s) and not _QUESTIONISH.search(s)]
    if _MY_FAVOURITE.search(user_prompt):
        said = [s for s in said if re.search(r"favou?rite", s, re.I)]
    stated = next((names for names in map(_real_names, reversed(said)) if names), [])
    if stated:
        names = " and ".join(dict.fromkeys(stated))
        if _MY_FAVOURITE.search(user_prompt):
            return "", _MY_FAVOURITE.sub(lambda _: names, user_prompt, count=1)
        return "", f"{user_prompt} ({names})"
    lines = [f"- Earlier in this chat, the user said: {s}" for s in said[-3:]]
    return ("From earlier in this conversation (chat recall):\n" + "\n".join(lines)
            if lines else ""), user_prompt


def _chat_recall(user_prompt: str, history: list[Message] | None) -> tuple[str, str]:
    """Resolve what the question refers back to in the chat.

    Returns ``(block, question)``. ``question`` is ``user_prompt`` with its
    back-reference written out — "What teams did he race for?" becomes "What
    teams did Nigel Mansell race for?" — or ``user_prompt`` itself when it has
    none. ``block`` is set only when the user points at their own earlier
    statements and no name can be read from them: those statements, labeled,
    to go into the prompt. (References to the chat's topic: ``dialogue.py``.)
    """
    if not history:
        return "", user_prompt
    return _own_words(user_prompt, history) or ("", references(user_prompt, follow(history)))


def _year_followup(user_prompt: str, history: list[Message] | None) -> str | None:
    """"And in 2007?" as the last full question with that year swapped in
    (see :func:`dialogue.followup`, which also swaps drivers, teams and races)."""
    return followup(user_prompt, follow(history)) if history else None


# --- deciding what a message needs --------------------------------------------- #
# "Who are you and what can you do?" got a paragraph about loving wet-weather
# qualifying, cut off mid-sentence (measured 2026-10-11). What this app is and
# does is not for the model to improvise.
_ABOUT_ME = re.compile(
    r"\b(?:who|what) (?:are|r) (?:you|u)(?=\W*$|\W+(?:and|then|exactly|really)\b)"
    r"|\bwhat(?:'s| is) your name\b"
    r"|\bwhat can you (?:do|help)\b|\bwhat (?:can|should|do) i ask\b"
    r"|\bare you (?:a |an )?(?:bot|robot|ai|human|real|chatbot|person|llm|model)\b"
    r"|\bhow do you work\b|^\W*(?:help|what do you (?:do|know))\W*$", re.I)
_ASKS = re.compile(r"\?|\b(who|what|when|where|which|why|how|explain|tell me)\b", re.I)


def _about_me() -> str:
    from pitgpt.rag.retriever import _get_entities

    entities = _get_entities()
    years = [int(race.year) for race, _ in entities.races] if entities else []
    span = f" My notes run from {min(years)} to the {max(years)} season, and when" if years else " When"
    return ("I'm PitGPT — a Formula 1 race engineer that runs entirely on this machine. "
            "Ask me who won a Grand Prix or a championship, about a driver's career, "
            "who drives for a team, a circuit, or how something like DRS works."
            f"{span} something isn't in them I'll say so rather than guess.")


@dataclass
class _Turn:
    """What one user message needs."""
    reply: str | None = None            # already determined: nothing is generated
    prefix: str = ""                    # said before the generated part
    plan: answers.Plan | None = None    # a grounded answer: facts, question, check
    question: str = ""                  # otherwise chat: what the model is asked...
    history: list[Message] | None = None    # ...with these earlier turns
    context: str = ""                   # chat-recall lines for the prompt


def _prepare(user_prompt, history: list[Message] | None, use_rag: bool = True) -> _Turn:
    """Work out what ``user_prompt`` needs, given the chat so far.

    The adapter is trained single-turn: with chat history in the prompt, the
    model imitates its own prior formatted answers instead of reading the
    fresh context (measured live: "Rosberg is the 2017 champion" after a
    correct 2016 answer, and the name-blend "Nico Mansell"). Grounded turns
    therefore answer FRESH from the retrieved facts, on a question that has
    been made to stand on its own; raw history is kept for persona/chat
    turns, where continuity is the point.
    """
    user_prompt = _content_to_text(user_prompt).strip()
    # Radio log outranks everything: "what is my favourite corn drink?" is
    # off-topic AND self-referential — the log must answer it.
    log_reply = _radio_log_reply(user_prompt, history)
    if log_reply is not None:
        return _Turn(reply=log_reply)
    prefix = ""
    fav = _favourite_lookup(user_prompt, history)
    if fav is not None:
        prefix, rewritten = fav
        if rewritten is None:
            return _Turn(reply=prefix)
        user_prompt = rewritten
    if not use_rag:
        return _Turn(prefix=prefix, question=user_prompt, history=history)

    from pitgpt.rag import retriever as R

    if R._get_retriever() is None:      # no index built: plain chat still works
        return _Turn(prefix=prefix, question=user_prompt, history=_last_exchanges(history))
    if _ABOUT_ME.search(user_prompt) and not R.has_f1_content(user_prompt):
        return _Turn(reply=_about_me())
    # The question as it stands on its own: drivers by full name, follow-ups
    # and back-references written out — for retrieval AND for the model.
    question = resolve(user_prompt, follow(history))
    recall, question = _own_words(question, history) or ("", question)
    found = R.find(question)
    if found.intent == "schedule":
        latest = R._latest_race(R._get_entities())
        race = parse_race(latest["text"]) if latest else None
        return _Turn(reply=T.no_live_data_reply(f"{race.year} {race.gp}" if race else ""))
    if found.intent == "off_topic":
        return _Turn(reply=T.OFF_TOPIC)
    if found.facts and found.facts[0].get("source") == "absence_guard":
        # The guards answer for themselves: the adapter predates them.
        guard = found.facts[0]
        return _Turn(reply=prefix + (guard.get("reply")
                                     or T.absence_reply(guard["year"], guard["gp"])))
    if found.facts:
        plan = answers.plan(found.question, found.facts, found.intent)
        if plan.settled:
            return _Turn(reply=prefix + plan.reply, plan=plan)
        if plan.lead:
            prefix += plan.lead + " "
        return _Turn(prefix=prefix, context=recall, plan=plan)
    if recall:
        return _Turn(prefix=prefix, question=question, context=recall)
    if (not R._is_conversational(found.question)
            and R.has_f1_content(found.question) and _ASKS.search(found.question)):
        # A Formula 1 question the notes hold nothing on. Left to free chat it
        # was answered from the transcript — "Who was on pole?" got a name out
        # of an earlier reply (measured 2026-10-11).
        return _Turn(reply=prefix + T.DECLINE)
    # Small talk: the persona, with the last few turns for continuity. The
    # whole transcript is no help: after a run of race recaps "Who drives for
    # Ferrari?" came back as a recap-shaped line about Carlos Sainz.
    return _Turn(prefix=prefix, question=user_prompt, history=_last_exchanges(history))


def _last_exchanges(history: list[Message] | None, n: int = 3) -> list[Message]:
    """The last ``n`` exchanges of ``history`` that were small talk themselves,
    as complete pairs (Gemma's chat template rejects turns that do not
    alternate).

    Grounded exchanges are left out: with them in the prompt the persona
    imitates their formats. Measured 2026-10-11, straight after a line-up
    answer and a decline: "What is the capital of France?" -> "Paris, my
    mistake — I didn't cover that one! What I helped you with is 'Lewis
    Hamilton and Charles Leclerc', nothing else."
    """
    from pitgpt.rag.retriever import _is_conversational, has_f1_content

    pairs: list[Message] = []
    turns = [t for t in history or [] if _content_to_text(t.get("content"))]
    for user, bot in zip(turns, turns[1:]):
        if user.get("role") == "user" and bot.get("role") == "assistant":
            said = _content_to_text(user.get("content"))
            if _is_conversational(said) or not has_f1_content(said):
                pairs += [user, bot]
    return pairs[-2 * n:]


def _grounding(turn: _Turn) -> str:
    """The context block for a turn: its facts, then any chat-recall lines."""
    from pitgpt.rag.retriever import format_context

    facts = format_context(turn.plan.facts) if turn.plan else ""
    return "\n\n".join(part for part in (facts, turn.context) if part)


# Ungrounded chat may not state figures: the system prompt asks the model not
# to cite years or statistics from memory, and it does anyway ("It came back
# to this three times between 1993 and 1997", about Senna and Prost).
_SENTENCES = re.compile(r".+?(?:[.!?…]+[\"'”’)\]]*(?=\s|$)|$)", re.S)
_FIGURE = re.compile(r"\d")


def _tidy_chat(text: str, done: bool = True) -> str:
    """Free-chat text fit to show: whole sentences only, none with a figure.

    The token budget ends chat replies mid-sentence ("...Do you prefer");
    ``done=False`` (still streaming) holds the sentence in progress back.
    """
    sentences = [s.strip() for s in _SENTENCES.findall(text.strip()) if s.strip()]
    if sentences and not re.search(r"[.!?…][\"'”’)\]]*$", sentences[-1]):
        if not done or len(sentences) > 1:
            sentences.pop()              # unfinished
    kept = [s for s in sentences if not _FIGURE.search(s)]
    return " ".join(kept or ([] if not done else sentences))


# A checked grounded reply is a function of the question put to the model
# and its facts (decoding is greedy), so the second time round it need not be
# generated: "Who was on pole?" and "Who finished second?" about one race are
# the same recap, eleven seconds apart on this laptop.
_checked: dict[tuple[str, str], tuple[str, str]] = {}


def _recall_checked(asked: str, grounding: str) -> tuple[str, str] | None:
    return None if settings.generation.do_sample else _checked.get((asked, grounding))


def _remember_checked(asked: str, grounding: str, reply: str, verdict: str) -> None:
    if len(_checked) >= 256:
        _checked.clear()
    _checked[(asked, grounding)] = (reply, verdict)




class _Halt(StoppingCriteria):
    """Stop generating once the reply has been decided without it."""

    def __init__(self) -> None:
        self.event = Event()

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(),
                          dtype=torch.bool, device=input_ids.device)


@torch.inference_mode()
def _generate(model, tokenizer, messages: list[Message], grounded: bool) -> str:
    """The model's reply to ``messages``, whole."""
    inputs = _encode(tokenizer, messages, model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        **_generation_kwargs(grounded=grounded),
    )
    # Only decode the newly generated tokens, not the prompt we fed in.
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def _generate_stream(model, tokenizer, messages: list[Message], grounded: bool) -> Iterator[str]:
    """The model's reply to ``messages``, piece by piece as it is generated.

    ``generate()`` runs on a background thread and its text is pulled off a
    ``TextIteratorStreamer`` queue. Closing this iterator early stops the
    generation and waits for the thread, so the next turn finds the GPU free.
    """
    inputs = _encode(tokenizer, messages, model.device)
    # The timeout only matters if generate() dies: without it the loop below
    # would wait on a queue nobody is filling.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                    timeout=180)
    halt = _Halt()
    thread = Thread(target=model.generate, kwargs={
        **inputs,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        "stopping_criteria": StoppingCriteriaList([halt]),
        **_generation_kwargs(grounded=grounded),
    })
    thread.start()
    drained = False
    try:
        for piece in streamer:
            yield piece
        drained = True
    finally:
        halt.event.set()
        if not drained:
            for _ in streamer:
                pass
        thread.join()


def _facts_of(context: str) -> list[dict]:
    """The fact lines of a ready-made context block (the eval harness's)."""
    return [{"text": line[2:].strip()} for line in context.splitlines()
            if line.startswith("- ")]


def generate_reply(
    model,
    tokenizer,
    user_prompt: str,
    history: list[Message] | None = None,
    use_rag: bool = True,
    context: str | None = None,
    check: bool | None = None,
    trace: dict | None = None,
) -> str:
    """Generate a single, complete enthusiast reply (blocking), grounded via RAG.

    ``context`` (if given) is used verbatim instead of live retrieval — this is
    how the evaluation harness injects *gold* context to measure generation
    grounding in isolation from retrieval quality. ``check`` says whether the
    reply is checked against its facts before it is returned: by default it is
    for live retrieval and is not for a given ``context``, so the grounding
    benchmark measures the model alone unless asked otherwise.

    ``trace`` (if given) is filled with what happened: the plan, the question
    put to the model, its own reply, and the verdict on it.
    """
    trace = trace if trace is not None else {}
    if context is not None:
        # The harness's question, its context, and nothing of ours in between.
        turn = _Turn(question=user_prompt, history=None if context else history, context=context)
        if context and check:
            turn.plan = answers.plan(user_prompt, _facts_of(context))
            turn.plan.ask = user_prompt
        grounding = context
    else:
        turn = _prepare(user_prompt, history, use_rag)
        check = True if check is None else check
        grounding = _grounding(turn)
    trace.update(reply=turn.reply, plan=turn.plan)
    if turn.reply is not None:
        return turn.reply

    plan = turn.plan if check else None
    asked = turn.plan.ask if turn.plan and context is None else turn.question
    known = _recall_checked(asked, grounding) if plan and context is None else None
    if known:
        trace.update(asked=asked, raw=None, verdict=known[1])
        return turn.prefix + known[0]
    messages = build_messages(asked, None if grounding else turn.history, context=grounding)
    raw = _generate(model, tokenizer, messages, grounded=bool(grounding))
    trace.update(asked=asked, raw=raw)
    if plan:
        raw, trace["verdict"] = answers.check(raw, plan)
        logger.debug("asked %r -> %s", asked, trace["verdict"])
        if context is None:
            _remember_checked(asked, grounding, raw, trace["verdict"])
    elif not grounding and (context is None or check):
        raw = _tidy_chat(raw)
    return turn.prefix + raw


def stream_reply(
    model,
    tokenizer,
    user_prompt: str,
    history: list[Message] | None = None,
    use_rag: bool = True,
    trace: dict | None = None,
) -> Iterator[str]:
    """Yield the reply as a growing string (for the UI), RAG-grounded.

    What is yielded is always fit to stay on screen: free chat a whole
    sentence at a time; a grounded reply while it is word for word what its
    fact dictates, or sentence by sentence as each passes the check
    (``answers.py``). A reply that can no longer pass is not waited for.
    """
    trace = trace if trace is not None else {}
    turn = _prepare(user_prompt, history, use_rag)
    trace.update(reply=turn.reply, plan=turn.plan)
    if turn.reply is not None:
        yield turn.reply
        return
    if turn.prefix:
        yield turn.prefix

    grounding = _grounding(turn)
    plan = turn.plan
    asked = plan.ask if plan else turn.question
    known = _recall_checked(asked, grounding) if plan else None
    if known:
        trace.update(asked=asked, raw=None, verdict=known[1])
        yield turn.prefix + known[0]
        return
    messages = build_messages(asked, None if grounding else turn.history, context=grounding)
    pieces = _generate_stream(model, tokenizer, messages, grounded=bool(grounding))

    raw = shown = ""
    try:
        for piece in pieces:
            raw += piece
            if plan is None:
                text = raw if grounding else _tidy_chat(raw, done=False)
            elif answers.hopeless(raw, plan) or answers.declined(raw, plan):
                # Nothing it goes on to write will be shown — it can no
                # longer pass the check, or it has declined and that is the
                # whole reply: stop generating, and show the verdict (below)
                # now rather than when it would have finished.
                break
            elif plan.strict:
                text = raw                    # word for word the dictated reply so far
            elif re.search(r"[.!?]", piece):
                # Worded freely: shown a sentence at a time, each only once
                # it has passed the check.
                text = answers.presentable(raw, plan) or shown
            else:
                continue
            if text != shown:
                shown = text
                yield turn.prefix + shown
    finally:
        pieces.close()                        # stops the generation if it is still going

    trace.update(asked=asked, raw=raw.strip())
    if plan is not None:
        final, trace["verdict"] = answers.check(raw, plan)
        logger.debug("asked %r -> %s", asked, trace["verdict"])
        _remember_checked(asked, grounding, final, trace["verdict"])
    else:
        final = raw.strip() if grounding else _tidy_chat(raw)
    if final != shown:
        yield turn.prefix + final


def batch_replies(
    model,
    tokenizer,
    prompts: Iterable[str],
) -> list[str]:
    """Generate one reply per prompt sequentially. Used by the eval harness."""
    return [generate_reply(model, tokenizer, p) for p in prompts]
