#!/usr/bin/env python3
"""Benchmark VLM morphological extraction across models and prompts.

This script enables systematic testing of Stage 1 (VLM morphology extraction)
across different vision models and prompt styles, with support for multi-image
input per specimen. Matches workshop_pipeline/step_1/predict.py functionality.

Usage:
    # List available prompts
    python tests/benchmarks/run_vlm_stage1.py --list-prompts

    # Run benchmark with sample JSON (multi-image per specimen)
    python tests/benchmarks/run_vlm_stage1.py --samples tests/benchmarks/configs/stage1_samples.json \
        --model Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 --prompt sys4 --port 8000

    # Run with few-shot examples
    python tests/benchmarks/run_vlm_stage1.py --samples tests/benchmarks/configs/stage1_samples.json \
        --model <MODEL> --prompt sys1 --examples tests/benchmarks/configs/examples.json

    # Generate HTML comparison report
    python tests/benchmarks/run_vlm_stage1.py --report results/vlm_benchmark/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import logging

import pandas as pd

from seedlearn.components.analyzers.prompts import (
    PromptStyle,
    get_prompt,
    is_multi_image_style,
    list_prompts as get_prompt_descriptions,
)
from seedlearn.pipeline.vlm_client import (
    InferenceClient as VLLMClient,
    InferenceConfig as VLLMConfig,
    _get_image_url,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path("results/vlm_benchmark")

# Expected keys for parsing form responses (24 morphological traits)
EXPECTED_KEYS = [
    "leaf relative position",
    "leaf spacing",
    "leaf complexity",
    "compound leaf type, only if leaf complexity is compound",
    "number of leaflets",
    "leaflet arrangement",
    "leaf margin",
    "leaf shape",
    "leaf apex",
    "leaf base",
    "venation type",
    "secondary veins",
    "leaf surface features",
    "leaf surface trichomes",
    "petiole length",
    "petiole features",
    "stem type, may not be visible",
    "stem trichomes",
    "stem color",
    "stem texture",
    "stipules, sometimes visible only in close-up images",
    "latex, rarely visible but extremely diagnostic when present",
    "pulvinus, for fabaceae family only",
    "tendrils",
    "notes",
]


# Mapping from form-parsed keys to scorer TRAIT_COLUMNS.
# Keys must match what parse_answer() produces — the VLM reproduces the full
# form line including parenthetical hints, so keys include those hints.
# We map both the short form (VLM omits hints) and full form (common case).
FORM_KEY_TO_TRAIT: dict[str, str] = {
    # Short keys (VLM omits parenthetical hints)
    "leaf complexity": "leaf_complexity",
    "leaf relative position": "leaf_arrangement",
    "leaf margin": "leaf_margin",
    "stipules, sometimes visible only in close-up images": "stipules",
    "latex, rarely visible but extremely diagnostic when present": "latex",
    # Full keys (VLM reproduces parenthetical hints — the common case)
    "leaf complexity (simple / compound)": "leaf_complexity",
    "leaf relative position (alternate / opposite / whorled)": "leaf_arrangement",
    "leaf margin (entire / toothed) and if toothed (dentate, serrate, etc.))": "leaf_margin",
    "stipules, sometimes visible only in close-up images (present / absent)": "stipules",
    "latex, rarely visible but extremely diagnostic when present (present / not observed)": "latex",
}


def write_config_json(
    output_dir: Path,
    model: str,
    prompt_style: str,
    mode: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    num_samples: int,
    num_examples: int,
) -> None:
    """Write run metadata to config.json in the output directory.

    Args:
        output_dir: Root output directory for the run.
        model: Model name/path.
        prompt_style: Prompt style used.
        mode: Inference mode (multi/single/both).
        temperature: Sampling temperature.
        max_tokens: Max generation tokens.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability parameter.
        num_samples: Number of specimens.
        num_examples: Number of few-shot examples.
    """
    config = {
        "model": model,
        "prompt_style": prompt_style,
        "mode": mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "num_samples": num_samples,
        "num_examples": num_examples,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info("Config written: %s", config_path)


def run_single_benchmark(
    samples: dict[str, list[str]],
    model: str,
    vlm_url: str,
    output_dir: Path,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = -1.0,
    examples: list[tuple[list[str], str]] | None = None,
    prompt_file: str | None = None,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Run single-image benchmark: each image processed independently.

    Uses SYS4_SINGLE prompt by default, or a custom prompt file if provided.
    Results are saved under ``output_dir/single/{specimen_key}/img_NNN.json``.

    Args:
        samples: Dictionary mapping specimen keys to image path lists.
        model: Model name/path.
        vlm_url: vLLM server URL.
        output_dir: Root output directory (single/ subdirectory will be created).
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability parameter.
        examples: Optional few-shot examples.
        prompt_file: Path to custom prompt text file (overrides default SYS4_SINGLE).

    Returns:
        Summary dictionary with per-image results.
    """
    single_dir = output_dir / "single"
    single_dir.mkdir(parents=True, exist_ok=True)

    # Use custom prompt file if provided, otherwise default to SYS4_SINGLE
    prompt_style = PromptStyle.SYS4_SINGLE
    if prompt_file:
        from seedlearn.pipeline.config import load_prompt

        prompt_text = load_prompt(prompt_file, get_prompt(prompt_style))
        logger.info(f"Single mode: using custom prompt from {prompt_file}")
    else:
        prompt_text = get_prompt(prompt_style)

    vllm_config = VLLMConfig(
        base_url=vlm_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    total_images = sum(len(paths) for paths in samples.values())

    print(f"\n{'=' * 60}")
    print(f"Single-Image Benchmark: {model}")
    print(f"Prompt: {prompt_style.value}")
    print(f"Specimens: {len(samples)} ({total_images} total images)")
    print(f"Output: {single_dir}")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    results_summary: dict[str, list[dict[str, Any]]] = {}

    # Build flat list of (specimen_key, img_idx, img_path) tasks
    tasks: list[tuple[str, int, str]] = []
    for specimen_key, image_paths in sorted(samples.items()):
        specimen_dir = single_dir / specimen_key
        specimen_dir.mkdir(parents=True, exist_ok=True)
        results_summary[specimen_key] = []
        for img_idx, img_path in enumerate(image_paths):
            tasks.append((specimen_key, img_idx, img_path))

    def _process_single_image(
        task_idx: int,
        specimen_key: str,
        img_idx: int,
        img_path: str,
    ) -> dict[str, Any] | None:
        """Process one image and return the result entry."""
        image_id = f"img_{img_idx:03d}"
        try:
            sample_start = time.perf_counter()

            # Each thread gets its own client to avoid shared state
            thread_client = VLLMClient(vllm_config)
            messages = build_messages_with_examples(
                sys_prompt=prompt_text,
                user_prompt=None,
                examples=examples,
                image_paths=[img_path],
                client=thread_client,
            )

            response = thread_client.chat(messages, strip_thinking=False)
            raw_content = response.raw_content
            thinking, answer = process_result(raw_content)

            parsed = parse_answer(answer)
            traits: dict[str, str] = {}
            for form_key, trait_col in FORM_KEY_TO_TRAIT.items():
                if form_key in parsed:
                    traits[trait_col] = parsed[form_key]

            elapsed = (time.perf_counter() - sample_start) * 1000

            result_entry = {
                "specimen_key": specimen_key,
                "image_id": image_id,
                "image_path": img_path,
                "traits": traits,
                "raw_response": answer,
                "thinking": thinking,
            }

            # Save individual result JSON
            result_path = single_dir / specimen_key / f"{image_id}.json"
            with open(result_path, "w") as f:
                json.dump(result_entry, f, indent=4)

            tokens = response.usage.get("total_tokens", "?")
            print(
                f"[{task_idx + 1}/{len(tasks)}] {specimen_key}/{image_id} "
                f"ok ({elapsed:.0f}ms, {tokens} tokens)"
            )
            return result_entry

        except Exception as e:
            logger.error("Failed %s/%s: %s", specimen_key, image_id, e)
            print(f"[{task_idx + 1}/{len(tasks)}] {specimen_key}/{image_id} FAIL: {e}")
            return None

    # Run with thread pool — vLLM handles concurrent requests via continuous batching
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import subprocess
    import threading

    max_workers = min(max_workers, len(tasks))
    print(f"Processing {len(tasks)} images with {max_workers} concurrent workers\n")

    # GPU utilization tracking
    gpu_samples: list[tuple[float, int]] = []  # (timestamp, utilization%)

    def _gpu_monitor(stop_event: threading.Event) -> None:
        """Sample GPU utilization every 2 seconds in a background thread."""
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    util = int(result.stdout.strip().split("\n")[0])
                    gpu_samples.append((time.time(), util))
            except Exception:
                pass
            stop_event.wait(2.0)

    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=_gpu_monitor, args=(stop_monitor,), daemon=True
    )
    monitor_thread.start()

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task_idx, (specimen_key, img_idx, img_path) in enumerate(tasks):
            future = executor.submit(
                _process_single_image, task_idx, specimen_key, img_idx, img_path
            )
            futures[future] = (specimen_key, img_idx)

        for future in as_completed(futures):
            specimen_key, img_idx = futures[future]
            result = future.result()
            if result:
                results_summary[specimen_key].append(result)

    stop_monitor.set()
    monitor_thread.join(timeout=3)

    # Sort per-specimen results by image_id for consistency
    for key in results_summary:
        results_summary[key].sort(key=lambda r: r.get("image_id", ""))

    total_time = (time.time() - start_time) / 60.0
    total_completed = sum(len(v) for v in results_summary.values())

    # Throughput report
    print(f"\n{'=' * 60}")
    print(
        f"Single-image complete: {total_completed}/{len(tasks)} images in {total_time:.2f} minutes"
    )
    if total_time > 0:
        imgs_per_min = total_completed / total_time
        print(f"Throughput: {imgs_per_min:.1f} images/min ({max_workers} workers)")
    if gpu_samples:
        avg_util = sum(u for _, u in gpu_samples) / len(gpu_samples)
        max_util = max(u for _, u in gpu_samples)
        min_util = min(u for _, u in gpu_samples)
        print(f"GPU utilization: avg={avg_util:.0f}%, min={min_util}%, max={max_util}%")
        if avg_util < 80:
            print(
                f"  Tip: GPU underutilized — try increasing --workers (current: {max_workers})"
            )
    print(f"{'=' * 60}\n")

    return {
        "mode": "single",
        "prompt_style": prompt_style.value,
        "total_images": total_completed,
        "total_runtime_min": round(total_time, 2),
        "specimens": list(results_summary.keys()),
    }


def list_prompts() -> None:
    """Print available prompt styles."""
    print("\n=== Available Prompt Styles ===\n")
    for style, desc in get_prompt_descriptions().items():
        multi = " [MULTI-IMAGE]" if is_multi_image_style(style) else ""
        print(f"  {style:6s} - {desc}{multi}")
    print()


def load_samples_json(filepath: str) -> dict[str, list[str]]:
    """Load multi-image samples from JSON file.

    Args:
        filepath: Path to JSON file with format:
            {"sample_id": ["image1.jpg", "image2.jpg", ...], ...}

    Returns:
        Dictionary mapping sample IDs to image path lists.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Samples file not found: {filepath}")

    with open(path) as f:
        samples = json.load(f)

    # Validate paths exist
    valid_samples = {}
    for sample_id, image_paths in samples.items():
        valid_paths = []
        for img_path in image_paths:
            # Handle relative paths
            p = Path(img_path)
            if not p.is_absolute():
                # Try relative to seedlearn data
                p = Path(
                    "/nfs/roberts/project/pi_lsc4/shared/seedlearn"
                ) / img_path.lstrip("./")
            if p.exists():
                valid_paths.append(str(p))
            else:
                logger.warning(f"Image not found, skipping: {img_path}")

        if valid_paths:
            valid_samples[sample_id] = valid_paths
        else:
            logger.warning(f"No valid images for sample {sample_id}, skipping")

    return valid_samples


def load_examples_json(filepath: str) -> list[tuple[list[str], str]] | None:
    """Load few-shot examples from JSON file.

    Args:
        filepath: Path to JSON file with format:
            {"idx": {"img_list": ["path1", "path2"], "target": "response text"}, ...}

    Returns:
        List of (image_paths, target_response) tuples.
    """
    if not filepath:
        return None

    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Examples file not found: {filepath}")
        return None

    with open(path) as f:
        examples_dict = json.load(f)

    examples = []
    for idx, example in examples_dict.items():
        img_list = example.get("img_list", [])
        target = example.get("target", "")
        # Resolve paths
        resolved_paths = []
        for img_path in img_list:
            p = Path(img_path)
            if not p.is_absolute():
                p = Path(
                    "/nfs/roberts/project/pi_lsc4/shared/seedlearn"
                ) / img_path.lstrip("./")
            if p.exists():
                resolved_paths.append(str(p))
        if resolved_paths and target:
            examples.append((resolved_paths, target))

    return examples if examples else None


def check_vllm_server(base_url: str) -> bool:
    """Check if vLLM server is accessible."""
    try:
        client = VLLMClient(VLLMConfig(base_url=base_url))
        return client.health_check()
    except Exception:
        return False


def process_result(output: str) -> tuple[str, str]:
    """Process raw VLM output to extract thinking and answer.

    Args:
        output: Raw VLM response text.

    Returns:
        Tuple of (thinking, answer).
    """
    thinking = ""
    answer = output

    # Extract thinking block
    if "</think>" in output:
        parts = output.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1]

    # Remove trailing #### marker
    if "####" in answer:
        answer = answer.split("####")[0]

    return thinking.strip(), answer.strip()


def parse_answer(answer_string: str) -> dict[str, str]:
    """Parse an answer string by splitting on newlines and colons.

    Matches workshop_pipeline/step_1/predict.py parse_answer function.

    Args:
        answer_string: String containing key:value pairs separated by newlines.

    Returns:
        Dictionary with lowercase keys and corresponding values.
    """
    parsed = {}
    lines = answer_string.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ":" in line:
            # Split only on the first colon
            key, value = line.split(":", 1)
            # Remove numbering (e.g., "1. Leaf relative position" -> "leaf relative position")
            key = key.strip()
            if "." in key:
                key = key.split(".", 1)[1].strip()
            key = key.lower()
            value = value.strip().replace("[NOTE]", "").replace("[REPORT]", "").strip()
            parsed[key] = value

    return parsed


def process_answers_to_csv(
    answers: dict[str, str],
    images_dict: dict[str, list[str]],
    save_path: str,
) -> pd.DataFrame:
    """Process answers dictionary and save to CSV.

    Matches workshop_pipeline/step_1/predict.py process_answers_to_csv function.

    Args:
        answers: Dictionary of format {"identifier": "answer_string"}.
        images_dict: Dictionary of format {"identifier": ["path1", "path2"]}.
        save_path: Path for JSON file (will be modified to .csv).

    Returns:
        DataFrame with parsed results.
    """
    csv_path = save_path.replace(".json", ".csv")
    failure_path = save_path.replace(".json", "_failures.txt")

    # Initialize DataFrame
    df = pd.DataFrame()

    # Process each answer
    for identifier, answer_string in answers.items():
        try:
            # Parse the answer
            parsed = parse_answer(answer_string)

            # Check for extra keys
            extra_keys = set(parsed.keys()) - set(EXPECTED_KEYS)
            if extra_keys:
                logger.warning(f"Unexpected keys for {identifier}: {extra_keys}")

            # Fill in missing keys with empty string
            for key in EXPECTED_KEYS:
                if key not in parsed:
                    parsed[key] = ""

            # Add image paths metadata
            parsed["image_paths"] = str(images_dict.get(identifier, []))

            # Add to DataFrame
            df[identifier] = pd.Series(parsed)

        except Exception as e:
            logger.error(f"Error processing answer for `{identifier}`: {e}")
            with open(failure_path, "a") as f:
                f.write(f"{identifier}\n")

    # Reorder rows: image_paths first, then traits
    if not df.empty:
        other_rows = [idx for idx in df.index if idx != "image_paths"]
        df = df.reindex(["image_paths"] + other_rows)
        df = df.T
        df.index.name = "specimen"
        df.to_csv(csv_path)
        print(f"CSV saved: {csv_path}")

    return df


def build_messages_with_examples(
    sys_prompt: str,
    user_prompt: str | None,
    examples: list[tuple[list[str], str]] | None,
    image_paths: list[str],
    client: VLLMClient,
) -> list[dict[str, Any]]:
    """Build messages array with optional few-shot examples.

    Args:
        sys_prompt: System prompt text.
        user_prompt: Optional user prompt text.
        examples: Optional list of (image_paths, target) tuples.
        image_paths: Image paths for the current query.
        client: VLLMClient for building image content.

    Returns:
        Messages array for OpenAI API.
    """
    messages = []

    image_mode = client.config.image_mode

    # Add few-shot examples first
    if examples:
        for example_images, example_target in examples:
            # System prompt for example
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt}],
                }
            )
            # Example query
            content = []
            for img_path in example_images:
                img_url = _get_image_url(img_path, image_mode)
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            if user_prompt:
                content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": content})
            # Example response
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example_target}],
                }
            )

    # Add actual query
    messages.append(
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_prompt}],
        }
    )

    content = []
    for img_path in image_paths:
        img_url = _get_image_url(img_path, image_mode)
        content.append({"type": "image_url", "image_url": {"url": img_url}})
    if user_prompt:
        content.append({"type": "text", "text": user_prompt})

    messages.append({"role": "user", "content": content})

    return messages


def run_benchmark(
    samples: dict[str, list[str]],
    model: str,
    prompt_style: PromptStyle,
    vlm_url: str,
    output_dir: Path,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = -1.0,
    examples: list[tuple[list[str], str]] | None = None,
    save_name: str | None = None,
    no_timestamp: bool = False,
    prompt_file: str | None = None,
) -> dict[str, Any]:
    """Run benchmark on samples with specified model and prompt.

    Args:
        samples: Dictionary mapping sample IDs to image paths.
        model: Model name/path.
        prompt_style: Prompt style to use.
        vlm_url: vLLM server URL.
        output_dir: Directory to save results.
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability parameter.
        examples: Optional few-shot examples.
        save_name: Optional custom save name.
        no_timestamp: If True, don't add timestamp to filename.

    Returns:
        Benchmark results dictionary.
    """
    # Multi results go under multi/ subdirectory
    multi_dir = output_dir / "multi"
    multi_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    vllm_config = VLLMConfig(
        base_url=vlm_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )

    client = VLLMClient(vllm_config)

    # Use custom prompt file if provided, otherwise use registry
    if prompt_file:
        from seedlearn.pipeline.config import load_prompt

        prompt_text = load_prompt(prompt_file, get_prompt(prompt_style))
        logger.info(f"Multi mode: using custom prompt from {prompt_file}")
    else:
        prompt_text = get_prompt(prompt_style)

    # Determine save path (legacy aggregate file still written for compatibility)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_name:
        if no_timestamp:
            save_file = f"{save_name}_results.json"
        else:
            save_file = f"{save_name}_results_{timestamp}.json"
    else:
        model_safe = model.replace("/", "_")
        if no_timestamp:
            save_file = f"{model_safe}_{prompt_style.value}_results.json"
        else:
            save_file = f"{model_safe}_{prompt_style.value}_results_{timestamp}.json"

    save_path = output_dir / save_file

    print(f"\n{'=' * 60}")
    print(f"Multi-Image Benchmark: {model}")
    print(f"Prompt: {prompt_style.value}")
    print(f"Samples: {len(samples)}")
    print(f"Examples: {len(examples) if examples else 0}")
    print(f"Output: {multi_dir}")
    print(f"{'=' * 60}\n")

    # Run benchmark
    start_time = time.time()
    raw_results = {}
    processed_results = {}
    cots = {}
    answers = {}

    for i, (sample_id, image_paths) in enumerate(sorted(samples.items()), 1):
        num_images = len(image_paths)
        print(
            f"[{i}/{len(samples)}] {sample_id} ({num_images} images)...",
            end=" ",
            flush=True,
        )

        try:
            sample_start = time.perf_counter()

            # Build messages with examples
            messages = build_messages_with_examples(
                sys_prompt=prompt_text,
                user_prompt=None,
                examples=examples,
                image_paths=image_paths,
                client=client,
            )

            # Call API
            response = client.chat(messages, strip_thinking=False)
            raw_content = response.raw_content

            # Process result
            thinking, answer = process_result(raw_content)

            elapsed = (time.perf_counter() - sample_start) * 1000

            raw_results[sample_id] = raw_content
            processed_results[sample_id] = {"thinking": thinking, "answer": answer}
            cots[sample_id] = thinking
            answers[sample_id] = answer

            # Save per-specimen JSON in scorer-compatible format under multi/
            parsed = parse_answer(answer)
            traits: dict[str, str] = {}
            for form_key, trait_col in FORM_KEY_TO_TRAIT.items():
                if form_key in parsed:
                    traits[trait_col] = parsed[form_key]

            specimen_result = {
                "specimen_key": sample_id,
                "image_id": "multi",
                "traits": traits,
                "raw_response": answer,
                "thinking": thinking,
            }
            specimen_path = multi_dir / f"{sample_id}.json"
            with open(specimen_path, "w") as f:
                json.dump(specimen_result, f, indent=4)

            tokens = response.usage.get("total_tokens", "?")
            print(f"ok ({elapsed:.0f}ms, {tokens} tokens)")

        except Exception as e:
            logger.error(f"Failed to process {sample_id}: {e}")
            raw_results[sample_id] = f"ERROR: {e}"
            processed_results[sample_id] = {"thinking": "", "answer": ""}
            cots[sample_id] = ""
            answers[sample_id] = ""
            print(f"FAIL: {e}")

    total_time = (time.time() - start_time) / 60.0

    # Prepare legacy aggregate results
    results = {
        "model": model,
        "generation_kwargs": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
        },
        "prompt_style": prompt_style.value,
        "images": str(samples),
        "examples": str(examples) if examples else "",
        "total_runtime": f"{total_time:.2f} minutes",
        "num_samples": len(samples),
        "answers": answers,
        "cots": cots,
        "raw_results": raw_results,
        "processed_results": processed_results,
    }

    # Save legacy aggregate JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nLegacy JSON saved: {save_path}")

    # Process to CSV
    print("Extracting answers to CSV...")
    process_answers_to_csv(answers, samples, str(save_path))

    print(f"\n{'=' * 60}")
    print(f"Complete: {len(samples)} samples in {total_time:.2f} minutes")
    print(f"{'=' * 60}\n")

    return results


def generate_html_report(results_dir: Path, output_file: Path | None = None) -> str:
    """Generate HTML comparison report from benchmark results.

    Args:
        results_dir: Directory containing benchmark JSON files.
        output_file: Output HTML file path (optional).

    Returns:
        Path to generated HTML report.
    """
    result_files = sorted(results_dir.glob("*.json"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return ""

    all_results = []
    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
            data["_file"] = rf.name
            all_results.append(data)

    html = _generate_comparison_html(all_results)

    if output_file is None:
        output_file = (
            results_dir
            / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )

    with open(output_file, "w") as f:
        f.write(html)

    print(f"Report generated: {output_file}")
    return str(output_file)


def _generate_comparison_html(all_results: list[dict]) -> str:
    """Generate HTML content for comparison report."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>VLM Benchmark Comparison</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary h2 {{ margin-top: 0; }}
        table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }}
        th {{ background: #4a90d9; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f7ff; }}
        .success {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .response {{ font-family: monospace; font-size: 0.85em; white-space: pre-wrap; max-height: 400px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 4px; }}
        .thinking {{ font-size: 0.8em; color: #666; font-style: italic; background: #fff3cd; padding: 8px; border-radius: 4px; margin-bottom: 8px; max-height: 200px; overflow-y: auto; }}
        details {{ margin: 5px 0; }}
        summary {{ cursor: pointer; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>VLM Morphology Extraction Benchmark</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="summary">
        <h2>Runs Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Prompt</th>
                <th>Samples</th>
                <th>Runtime</th>
                <th>File</th>
            </tr>
"""

    for run in all_results:
        model = run.get("model", "N/A")
        model_short = model.split("/")[-1] if model else "N/A"
        html += f"""            <tr>
                <td title="{model}">{model_short}</td>
                <td>{run.get("prompt_style", "N/A")}</td>
                <td>{run.get("num_samples", len(run.get("answers", {})))}</td>
                <td>{run.get("total_runtime", "N/A")}</td>
                <td>{run.get("_file", "N/A")}</td>
            </tr>
"""

    html += """        </table>
    </div>

    <h2>Detailed Results by Sample</h2>
"""

    # Group by sample
    all_samples = set()
    for run in all_results:
        all_samples.update(run.get("answers", {}).keys())

    for sample_id in sorted(all_samples):
        html += f"""
    <details>
        <summary>{sample_id}</summary>
        <table>
            <tr>
                <th>Model</th>
                <th>Prompt</th>
                <th>Thinking</th>
                <th>Answer</th>
            </tr>
"""
        for run in all_results:
            model = run.get("model", "N/A")
            model_short = model.split("/")[-1][:25] if model else "N/A"
            prompt = run.get("prompt_style", "N/A")

            cot = run.get("cots", {}).get(sample_id, "")
            answer = run.get("answers", {}).get(sample_id, "")

            # Escape HTML
            cot_escaped = (
                cot.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            answer_escaped = (
                answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )

            html += f"""            <tr>
                <td>{model_short}</td>
                <td>{prompt}</td>
                <td><details><summary>Thinking ({len(cot)} chars)</summary><div class="thinking">{cot_escaped}</div></details></td>
                <td><div class="response">{answer_escaped}</div></td>
            </tr>
"""

        html += """        </table>
    </details>
"""

    html += """
</body>
</html>
"""

    return html


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark VLM morphological extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Actions
    parser.add_argument(
        "--list-prompts", action="store_true", help="List available prompt styles"
    )
    parser.add_argument(
        "--report",
        type=str,
        metavar="DIR",
        help="Generate HTML report from results directory",
    )

    # Benchmark options
    parser.add_argument(
        "--samples",
        "-s",
        type=str,
        help="Path to samples JSON file (multi-image per specimen)",
    )
    parser.add_argument("--model", "-m", type=str, help="Vision model name/path")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="sys1",
        choices=["sys1", "sys2", "sys3", "sys4", "sys4_single", "json"],
        help="Prompt style (default: sys1)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to custom prompt text file (overrides --prompt)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        choices=["multi", "single", "both"],
        help="Inference mode: multi (all images), single (per-image), both (default: multi)",
    )
    parser.add_argument(
        "--all-prompts", action="store_true", help="Run all prompt styles"
    )
    parser.add_argument(
        "--examples", "-e", type=str, help="Path to few-shot examples JSON file"
    )

    # Concurrency options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Concurrent workers for single-image mode (default: 8)",
    )

    # Server options
    parser.add_argument(
        "--port", type=int, default=8000, help="vLLM server port (default: 8000)"
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default=None,
        help="Full vLLM server URL (overrides --port)",
    )

    # Generation options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for generation (default: 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Top-k sampling (default: 20)"
    )
    parser.add_argument(
        "--min-p", type=float, default=-1.0, help="Min-p sampling (default: -1.0)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--save-name", type=str, help="Custom save name for results")
    parser.add_argument(
        "--no-timestamp", action="store_true", help="Don't add timestamp to filename"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Setup logging
    logging.getLogger().setLevel(logging.WARNING if args.quiet else logging.INFO)

    # Handle list prompts
    if args.list_prompts:
        list_prompts()
        return 0

    # Handle report generation
    if args.report:
        generate_html_report(Path(args.report))
        return 0

    # Validate benchmark args
    if not args.samples:
        parser.error(
            "--samples is required for benchmark (or use --list-prompts / --report)"
        )

    if not args.model:
        parser.error("--model is required for benchmark")

    # Determine vLLM URL
    vlm_url = args.vlm_url or f"http://localhost:{args.port}/v1"

    # Check server
    print(f"Checking vLLM server at {vlm_url}...")
    if not check_vllm_server(vlm_url):
        print(f"\n✗ vLLM server not accessible at {vlm_url}")
        print("\nStart server with:")
        print(
            f"  vllm serve {args.model} --dtype auto --trust-remote-code --port {args.port} --allowed-local-media-path /"
        )
        return 1
    print("✓ vLLM server connected\n")

    # Load samples
    try:
        samples = load_samples_json(args.samples)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if not samples:
        print("Error: No valid samples found")
        return 1

    print(f"Loaded {len(samples)} samples from {args.samples}")

    # Load examples
    examples = load_examples_json(args.examples) if args.examples else None
    if examples:
        print(f"Loaded {len(examples)} few-shot examples from {args.examples}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode

    # Resolve prompt: --prompt-file takes priority over --prompt
    prompt_file = args.prompt_file
    style = PromptStyle(args.prompt)
    if prompt_file:
        print(f"Using custom prompt file: {prompt_file}")

    # Write config.json with run metadata
    write_config_json(
        output_dir=output_dir,
        model=args.model,
        prompt_style=style.value if not prompt_file else f"file:{prompt_file}",
        mode=mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        num_samples=len(samples),
        num_examples=len(examples) if examples else 0,
    )

    # Run multi mode
    if mode in ("multi", "both"):
        if args.all_prompts:
            for ps in PromptStyle:
                run_benchmark(
                    samples=samples,
                    model=args.model,
                    prompt_style=ps,
                    vlm_url=vlm_url,
                    output_dir=output_dir,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    examples=examples,
                    save_name=args.save_name,
                    no_timestamp=args.no_timestamp,
                    prompt_file=prompt_file,
                )
        else:
            run_benchmark(
                samples=samples,
                model=args.model,
                prompt_style=style,
                vlm_url=vlm_url,
                output_dir=output_dir,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                examples=examples,
                save_name=args.save_name,
                no_timestamp=args.no_timestamp,
                prompt_file=prompt_file,
            )

    # Run single mode
    if mode in ("single", "both"):
        run_single_benchmark(
            samples=samples,
            model=args.model,
            vlm_url=vlm_url,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            examples=examples,
            prompt_file=prompt_file,
            max_workers=args.workers,
        )

    print("\nTo generate comparison report:")
    print(f"  python benchmarks/run_vlm_stage1.py --report {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
