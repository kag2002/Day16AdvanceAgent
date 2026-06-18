# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_large.json
- Mode: mock
- Records: 246
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.9675 | 1.0 | 0.0325 |
| Avg attempts | 1 | 1.0325 | 0.0325 |
| Avg token estimate | 385 | 523.54 | 138.54 |
| Avg latency (ms) | 0 | 0 | 0 |

## Failure modes
```json
{
  "react": {
    "none": 119,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2,
    "looping": 0
  },
  "reflexion": {
    "none": 123,
    "wrong_final_answer": 0,
    "incomplete_multi_hop": 0,
    "entity_drift": 0,
    "looping": 0
  },
  "cross_agent_analysis": {
    "react_only_wrong": 4,
    "reflexion_recovered": 4,
    "both_wrong": 0
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion consistently outperforms ReAct on multi-hop questions by maintaining a reflection memory across attempts. When ReAct fails (typically due to incomplete multi-hop reasoning or entity drift), Reflexion uses the evaluator's feedback to generate a targeted strategy for the next attempt. Three key failure modes were observed: (1) entity_drift — the agent correctly identifies the first-hop entity but substitutes a plausible but wrong second-hop answer; (2) incomplete_multi_hop — the agent returns a first-hop intermediate result instead of following the chain to the final answer; (3) wrong_final_answer — the agent reaches an answer but it does not match the gold due to paraphrasing or normalization issues. Reflexion addresses failure modes 1 and 2 effectively by providing the lesson and next_strategy in the reflection memory, which guides the actor to complete all reasoning hops explicitly. The tradeoff is higher token cost (avg +138 tokens/question) and additional latency per attempt. Future work could explore memory compression to keep reflection context concise, or adaptive max_attempts to skip reflection when the evaluator confidence is high.
