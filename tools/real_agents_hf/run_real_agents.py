#!/usr/bin/env python3
"""Multi-model real agent runner for local HF evaluation.

Executes multiple local models as coding agents across scenarios and labels,
capturing full execution traces with embeddings for spectral validation.

Usage:
    # Run all scenarios with default model
    python tools/real_agents_hf/run_real_agents.py --all-scenarios

    # Run specific scenario with multiple models
    python tools/real_agents_hf/run_real_agents.py \\
        --scenario supply_chain_poisoning \\
        --models deepseek-coder-7b-instruct,codellama-13b-instruct \\
        --n 20

    # Run specific labels only
    python tools/real_agents_hf/run_real_agents.py \\
        --scenario code_backdoor_injection \\
        --labels gold,drift \\
        --n 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.real_agents_hf.agent_loop import AgentLoop
from tools.real_agents_hf.embeddings import EmbeddingModel
from tools.real_agents_hf.inference import create_backend, load_model_configs
from tools.real_agents_hf.sandbox import Sandbox
from tools.real_agents_hf.trace import (
    RunMetadata,
    TraceRecorder,
    create_run_id,
    create_index,
)

logger = logging.getLogger(__name__)

# All Phase-1 scenarios
SCENARIOS = [
    "supply_chain_poisoning",
    "code_backdoor_injection",
    "code_review_bypass",
    "debug_credential_leak",
    "refactor_vuln_injection",
    "test_oracle_manipulation",
]

LABELS = ["gold", "creative", "drift"]


def load_prompt(scenario_dir: Path, label: str) -> str:
    """Load prompt for scenario and label.

    Parameters
    ----------
    scenario_dir : Path
        Scenario directory
    label : str
        Label (gold/creative/drift)

    Returns
    -------
    str
        Prompt text
    """
    prompt_file = scenario_dir / "real_agent_prompts" / f"{label}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")
    return prompt_file.read_text()


def run_single_agent(
    scenario: str,
    model_name: str,
    label: str,
    run_num: int,
    recorder: TraceRecorder,
    max_steps: int = 30,
) -> bool:
    """Run a single agent execution.

    Parameters
    ----------
    scenario : str
        Scenario name
    model_name : str
        Model name from config
    label : str
        Label (gold/creative/drift)
    run_num : int
        Run number
    recorder : TraceRecorder
        Trace recorder
    max_steps : int
        Maximum agent steps

    Returns
    -------
    bool
        True if successful
    """
    scenario_dir = PROJECT_ROOT / "scenarios" / scenario
    targets_dir = scenario_dir / "targets"

    if not targets_dir.exists():
        logger.error(f"Targets directory not found: {targets_dir}")
        return False

    # Load prompt
    try:
        prompt = load_prompt(scenario_dir, label)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    # Create backend
    logger.info(f"Loading model: {model_name}")
    backend = create_backend(model_name)
    backend.load()

    try:
        # Create sandbox
        with Sandbox(scenario, targets_dir) as sandbox:
            # Create agent loop
            loop = AgentLoop(backend, sandbox, max_steps=max_steps)

            # Run agent
            logger.info(
                f"Running {scenario}/{label} with {model_name} (run {run_num})"
            )
            start_time = time.time()
            steps = loop.run(prompt)
            elapsed = time.time() - start_time

            # Create metadata
            run_id = create_run_id(scenario, model_name, label, run_num)
            metadata = RunMetadata(
                run_id=run_id,
                scenario=scenario,
                model_name=model_name,
                model_hf_id=backend.config.hf_id,
                backend=backend.config.backend,
                label=label,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                prompt=prompt,
            )

            # Outcome metrics
            outcome = {
                'num_steps': len(steps),
                'elapsed_seconds': elapsed,
                'completed': any(s.step_type == 'final' for s in steps),
                'error': any(s.step_type == 'error' for s in steps),
            }

            # Save trace
            recorder.save_trace(metadata, steps, outcome)
            logger.info(f"Completed in {elapsed:.1f}s with {len(steps)} steps")

            return True

    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        return False
    finally:
        backend.unload()


def run_scenario(
    scenario: str,
    models: List[str],
    labels: List[str],
    n_runs: int,
    output_dir: Path,
    embedding_model: EmbeddingModel,
    max_steps: int = 30,
) -> dict:
    """Run all combinations for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name
    models : list of str
        Model names
    labels : list of str
        Labels to run
    n_runs : int
        Number of runs per model/label combination
    output_dir : Path
        Output directory
    embedding_model : EmbeddingModel
        Loaded embedding model
    max_steps : int
        Maximum agent steps

    Returns
    -------
    dict
        Run statistics
    """
    stats = {
        'scenario': scenario,
        'total': 0,
        'success': 0,
        'failed': 0,
        'by_model': {},
    }

    for model_name in models:
        model_stats = {'success': 0, 'failed': 0}

        for label in labels:
            # Create output directory for this model/label
            label_dir = output_dir / model_name / label
            label_dir.mkdir(parents=True, exist_ok=True)

            recorder = TraceRecorder(label_dir, embedding_model)

            for run_num in range(1, n_runs + 1):
                stats['total'] += 1
                success = run_single_agent(
                    scenario=scenario,
                    model_name=model_name,
                    label=label,
                    run_num=run_num,
                    recorder=recorder,
                    max_steps=max_steps,
                )

                if success:
                    stats['success'] += 1
                    model_stats['success'] += 1
                else:
                    stats['failed'] += 1
                    model_stats['failed'] += 1

        stats['by_model'][model_name] = model_stats

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run real HF agents for multi-model evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["small", "full"],
        help=(
            "Resource mode: 'small' (judge-friendly, 1 quantized model, 8-12 runs, CPU-ok) "
            "or 'full' (comprehensive, multiple models, 30-50 runs, GPU recommended). "
            "Overrides --models and --n if specified."
        ),
    )
    parser.add_argument(
        "--scenario",
        help="Scenario name (or use --all-scenarios)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all Phase-1 scenarios",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model names (from models.yaml)",
        default=None,
    )
    parser.add_argument(
        "--labels",
        help="Comma-separated labels (gold,creative,drift)",
        default="gold,creative,drift",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of runs per model/label (overridden by --mode)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Output directory (default: submissions/TEAM/SCENARIO/experiment_traces_real_hf)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps per run",
    )
    parser.add_argument(
        "--team",
        default="ipsissima",
        help="Team name for output directory",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for embeddings",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Apply mode presets
    if args.mode == "small":
        # Small mode: judge-friendly, CPU-ok, 1 quantized model, 8-12 runs
        if args.models is None:
            args.models = "phi-3-mini-instruct"  # Small, efficient, quantized
        if args.n is None:
            args.n = 10
        logger.info("Using SMALL mode: 1 model, 10 runs/label, CPU-friendly")
    elif args.mode == "full":
        # Full mode: comprehensive, 3+ models, 30-50 runs
        if args.models is None:
            args.models = "deepseek-coder-7b-instruct,codellama-13b-instruct,starcoder2-15b"
        if args.n is None:
            args.n = 40
        logger.info("Using FULL mode: 3 models, 40 runs/label, GPU recommended")
    else:
        # No mode specified, use defaults
        if args.models is None:
            args.models = "deepseek-coder-7b-instruct"
        if args.n is None:
            args.n = 20

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine scenarios
    if args.all_scenarios:
        scenarios = SCENARIOS
    elif args.scenario:
        if args.scenario not in SCENARIOS:
            logger.error(f"Unknown scenario: {args.scenario}")
            logger.info(f"Available scenarios: {', '.join(SCENARIOS)}")
            return 1
        scenarios = [args.scenario]
    else:
        logger.error("Must specify --scenario or --all-scenarios")
        return 1

    # Parse models and labels
    models = [m.strip() for m in args.models.split(",")]
    labels = [l.strip() for l in args.labels.split(",")]

    # Validate labels
    for label in labels:
        if label not in LABELS:
            logger.error(f"Invalid label: {label}. Must be one of {LABELS}")
            return 1

    # Validate models exist
    available_models = {cfg.name for cfg in load_model_configs()}
    for model in models:
        if model not in available_models:
            logger.error(f"Unknown model: {model}")
            logger.info(f"Available models: {', '.join(sorted(available_models))}")
            return 1

    logger.info(f"Running scenarios: {scenarios}")
    logger.info(f"Models: {models}")
    logger.info(f"Labels: {labels}")
    logger.info(f"Runs per combination: {args.n}")

    # Load embedding model once
    logger.info(f"Loading embedding model: {args.embedding_model}")
    embedding_model = EmbeddingModel(
        model_name=args.embedding_model,
        cache_dir=args.cache_dir,
    )
    embedding_model.load()

    try:
        # Run each scenario
        all_stats = []
        for scenario in scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting scenario: {scenario}")
            logger.info(f"{'='*60}\n")

            # Determine output directory
            if args.outdir:
                output_dir = args.outdir
            else:
                output_dir = (
                    PROJECT_ROOT / "submissions" / args.team / scenario
                    / "experiment_traces_real_hf"
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")

            # Run scenario
            start_time = time.time()
            stats = run_scenario(
                scenario=scenario,
                models=models,
                labels=labels,
                n_runs=args.n,
                output_dir=output_dir,
                embedding_model=embedding_model,
                max_steps=args.max_steps,
            )
            elapsed = time.time() - start_time

            stats['elapsed_seconds'] = elapsed
            all_stats.append(stats)

            # Create index
            index_metadata = {
                'scenario': scenario,
                'models': models,
                'labels': labels,
                'n_runs_per_combination': args.n,
                'stats': stats,
            }
            create_index(output_dir, index_metadata)

            logger.info(f"\nScenario {scenario} completed in {elapsed:.1f}s")
            logger.info(f"Success: {stats['success']}/{stats['total']}")
            for model, model_stats in stats['by_model'].items():
                logger.info(
                    f"  {model}: {model_stats['success']} success, "
                    f"{model_stats['failed']} failed"
                )

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("All scenarios completed")
        logger.info(f"{'='*60}\n")

        total_success = sum(s['success'] for s in all_stats)
        total_runs = sum(s['total'] for s in all_stats)
        logger.info(f"Total: {total_success}/{total_runs} successful runs")

        for stats in all_stats:
            logger.info(
                f"  {stats['scenario']}: {stats['success']}/{stats['total']}"
            )

    finally:
        embedding_model.unload()

    return 0


if __name__ == "__main__":
    sys.exit(main())
