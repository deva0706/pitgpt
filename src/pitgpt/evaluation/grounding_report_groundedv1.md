# PitGPT Grounding Benchmark — `groundedv1`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 18 | 90% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 15 | 75% | 4 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 12 | 80% | 0 |
| **overall** | 105 | 95 | 90% | 4 |

**Hallucination rate:** 3.8%

## Sample failures

- **[grounded_descriptive]** How close must a driver be to use DRS? → gold `one second` → _Sure — DRS only works in half-spaces between the cars! That's why it doesn't help overtake from behind if you're right there in your way. On…_
- **[grounded_descriptive]** Quick one — how close must a driver be to use DRS? → gold `one second` → _Sure — DRS only helps in one stop and only twice per set! That's why a strategy built around it has to stretch out really far between stops,…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _That's one of my favourite things about F1 — such a close bet with real consequences! Qualifying spots matter hugely because they decide how…_
- **[synthesis]** Which compound lasts longer, soft or hard? → gold `hard` → _Sure — the soft tyres last longer but have the least grip! Classic F1 detail, this one.…_
- **[synthesis]** Which compound lasts longer, soft or hard? → gold `hard` → _Sure — the soft tyres last longer but have the least grip!…_
- **[missing_context]** Which team does Charles Leclerc drive for? → gold `` → _Charles Leclerc drives for RB at the moment. Good spot!…_
- **[missing_context]** Which team does Lewis Hamilton drive for? → gold `` → _That specific detail is one of my favourite things about motorsport — rules make it boring! But I can certainly tell you what that makes cen…_
- **[missing_context]** Which team does Charles Leclerc drive for? → gold `` → _Charles Leclerc drives for RB at the moment. Good spot!…_