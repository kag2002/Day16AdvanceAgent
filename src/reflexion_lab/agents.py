from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Literal

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

# Runtime mode: "mock" uses deterministic mock, "llm" uses real Gemini API
RUNTIME_MODE: Literal["mock", "llm"] = "mock"

def _load_runtime():
    if RUNTIME_MODE == "llm":
        from . import llm_runtime as rt
    else:
        from . import mock_runtime as rt
    return rt

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        rt = _load_runtime()
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            t_start = time.perf_counter()
            answer = rt.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = rt.evaluator(example, answer)
            t_end = time.perf_counter()
            # TODO: Replace with actual token count from LLM response
            # (hiện tại dùng ước lượng; khi dùng LLM thật, lấy từ response.usage)
            token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
            # TODO: Replace with actual latency measurement
            # (đã đo bằng perf_counter; khi dùng LLM thật có thể giữ nguyên)
            latency_ms = int((t_end - t_start) * 1000)
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break

            # TODO: Học viên triển khai logic Reflexion tại đây
            # 1. Kiểm tra nếu agent_type là 'reflexion' và chưa hết số lần attempt
            # 2. Gọi hàm reflector để lấy nội dung reflection
            # 3. Cập nhật reflection_memory để Actor dùng cho lần sau
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = rt.reflector(example, attempt_id, judge)
                reflection_memory.append(
                    f"[Attempt {reflection.attempt_id}] Lesson: {reflection.lesson} "
                    f"| Strategy: {reflection.next_strategy}"
                )
                reflections.append(reflection)
                trace.reflection = reflection

            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        # For mock mode, use FAILURE_MODE_BY_QID; for llm mode, derive from score
        failure_mode_map = getattr(rt, "FAILURE_MODE_BY_QID", {})
        failure_mode = "none" if final_score == 1 else failure_mode_map.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
