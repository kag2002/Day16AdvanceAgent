from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import typer
import src.reflexion_lab.agents as agents_module
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)

def _log(msg: str) -> None:
    """Flush-safe print for both stdout and file logging."""
    print(msg, flush=True)
    sys.stdout.flush()

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = typer.Option("mock", help="Runtime mode: 'mock' or 'llm'"),
) -> None:
    # Set runtime mode before loading agents
    agents_module.RUNTIME_MODE = mode  # type: ignore[assignment]
    _log(f"Mode: {mode}  |  Dataset: {dataset}")

    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    # ── ReAct pass ────────────────────────────────────────────────────────────
    react_records = []
    _log(f"\n[ReAct] Running on {len(examples)} examples...")
    for i, example in enumerate(examples, 1):
        record = react.run(example)
        react_records.append(record)
        status = "CORRECT" if record.is_correct else "WRONG"
        _log(f"  [{i:3d}/{len(examples)}] {example.qid:10s} | {status} | pred={record.predicted_answer[:40]!r}")

    react_em = sum(1 for r in react_records if r.is_correct) / len(react_records)
    _log(f"[ReAct] Done — EM={react_em:.2%}")

    # ── Reflexion pass ────────────────────────────────────────────────────────
    reflexion_records = []
    _log(f"\n[Reflexion] Running on {len(examples)} examples (max {reflexion_attempts} attempts)...")
    for i, example in enumerate(examples, 1):
        record = reflexion.run(example)
        reflexion_records.append(record)
        status = "CORRECT" if record.is_correct else "WRONG"
        _log(f"  [{i:3d}/{len(examples)}] {example.qid:10s} | {status} | attempts={record.attempts} | pred={record.predicted_answer[:40]!r}")

    reflexion_em = sum(1 for r in reflexion_records if r.is_correct) / len(reflexion_records)
    _log(f"[Reflexion] Done — EM={reflexion_em:.2%}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    _log(f"\nSaved: {json_path}")
    _log(f"Saved: {md_path}")
    _log(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
