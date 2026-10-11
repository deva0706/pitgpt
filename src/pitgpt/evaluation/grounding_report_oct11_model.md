# PitGPT Grounding Benchmark — `oct11_model`

Model: `google/gemma-3-1b-it` + adapter `pitgpt-lora`
Cases: 105

| Category | Cases | Pass | Accuracy | Hallucinations |
|---|---|---|---|---|
| grounded_descriptive | 20 | 15 | 75% | 0 |
| grounded_entity | 40 | 40 | 100% | 0 |
| missing_context | 20 | 6 | 30% | 14 |
| persona | 10 | 10 | 100% | 0 |
| synthesis | 15 | 6 | 40% | 0 |
| **overall** | 105 | 77 | 73% | 14 |

**Hallucination rate:** 13.3%

## Sample failures

- **[grounded_descriptive]** When did F1 switch to turbo-hybrid engines? → gold `2014` → _It was turbocharged hybrid power! More than one place — we swapped out natural fuel injection for a complex system of twin turbines and an e…_
- **[grounded_descriptive]** Quick one — when did F1 switch to turbo-hybrid engines? → gold `2014` → _It was turbocharged hybrid time! Teams could use a small turbocharger to boost engine power while keeping emissions down, plus there were co…_
- **[grounded_descriptive]** Quick one — which championship did Nico Rosberg win? → gold `2016` → _Constructing a sandwich board — two slices of buttered bread, jam in between, maybe some fruit. Classic, reliable, and endlessly customizabl…_
- **[grounded_descriptive]** Which team was Kimi Raikkonen with when he won his title? → gold `Ferrari` → _Winning the title is as much about winning championships as it is anything else. That one tells us Kimi Raikkonen took it in 2007.…_
- **[grounded_descriptive]** Quick one — which team was Kimi Raikkonen with when he won his title? → gold `Ferrari` → _Winning a championship is such a thing we feel about every weekend! It means being here at the end, not last. That's why watching someone ru…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Stack them up: Juan Manuel Fangio takes it — five championships to four.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Stack them up: Fernando Alonso takes it — seven championships to six.…_
- **[synthesis]** Of these drivers, who won the most World Championships? → gold `Schumacher` → _Stack them up: Lewis Hamilton takes it — seven championships to five.…_