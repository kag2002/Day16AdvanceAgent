# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_golden.json
- Mode: llm
- Records: 40
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1.05 | 0.05 |
| Avg token estimate | 385 | 533.5 | 148.5 |
| Avg latency (ms) | 31546.4 | 104345.95 | 72799.55 |

## Failure modes
```json
{
  "react": {
    "none": 20
  },
  "reflexion": {
    "none": 20
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
