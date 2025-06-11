import argparse
import json
import random
from typing import Dict, Optional, List

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from geo_bot import GeoBot
from benchmark import MapGuesserBenchmark
from config import MODELS_CONFIG, DATA_PATHS, SUCCESS_THRESHOLD_KM


def agent_mode(model_name: str, steps: int, headless: bool, samples: int, temperature: float = 0.0):
    """
    Runs the AI Agent in a benchmark loop over multiple samples,
    using multi-step exploration for each.
    """
    print(
        f"Starting Agent Mode (as a benchmark): model={model_name}, steps={steps}, samples={samples}, temperature={temperature}"
    )

    try:
        with open(DATA_PATHS["golden_labels"], "r", encoding="utf-8") as f:
            golden_labels = json.load(f).get("samples", [])
    except FileNotFoundError:
        print(f"Error: Golden labels file not found at {DATA_PATHS['golden_labels']}.")
        return

    if not golden_labels:
        print("Error: No samples found in golden_labels.json.")
        return

    num_to_test = min(samples, len(golden_labels))
    test_samples = golden_labels[:num_to_test]
    print(f"Will run on {len(test_samples)} samples.")

    config = MODELS_CONFIG.get(model_name)
    model_class = globals()[config["class"]]
    model_instance_name = config["model_name"]

    benchmark_helper = MapGuesserBenchmark(headless=True)
    all_results = []

    with GeoBot(
        model=model_class, model_name=model_instance_name, headless=headless, temperature=temperature
    ) as bot:
        for i, sample in enumerate(test_samples):
            print(
                f"\n--- Running Sample {i + 1}/{len(test_samples)} (ID: {sample.get('id')}) ---"
            )

            # **FIXED**: Correct sequence: Load Data -> Clean Environment -> Run Loop
            if not bot.controller.load_location_from_data(sample):
                print(
                    f"   ❌ Failed to load location for sample {sample.get('id')}. Skipping."
                )
                continue

            bot.controller.setup_clean_environment()

            final_guess = bot.run_agent_loop(max_steps=steps)

            true_coords = {"lat": sample.get("lat"), "lng": sample.get("lng")}
            distance_km = None
            is_success = False

            if final_guess:
                distance_km = benchmark_helper.calculate_distance(
                    true_coords, final_guess
                )
                if distance_km is not None:
                    is_success = distance_km <= SUCCESS_THRESHOLD_KM

                print(f"\nResult for Sample ID: {sample.get('id')}")
                print(
                    f"  Ground Truth: Lat={true_coords['lat']:.4f}, Lon={true_coords['lng']:.4f}"
                )
                print(
                    f"  Final Guess:  Lat={final_guess[0]:.4f}, Lon={final_guess[1]:.4f}"
                )
                dist_str = f"{distance_km:.1f} km" if distance_km is not None else "N/A"
                print(f"  Distance: {dist_str}, Success: {is_success}")
            else:
                print("Agent did not make a final guess for this sample.")

            all_results.append(
                {
                    "sample_id": sample.get("id"),
                    "model": bot.model_name,
                    "true_coordinates": true_coords,
                    "predicted_coordinates": final_guess,
                    "distance_km": distance_km,
                    "success": is_success,
                }
            )

    summary = benchmark_helper.generate_summary(all_results)
    if summary:
        print("\n\n--- Agent Benchmark Complete! Summary ---")
        for model, stats in summary.items():
            print(f"Model: {model}")
            print(f"  Success Rate: {stats['success_rate'] * 100:.1f}%")
            print(f"  Avg Distance: {stats['average_distance_km']:.1f} km")

    print("\nAgent Mode finished.")


def benchmark_mode(models: list, samples: int, headless: bool, temperature: float = 0.0):
    """Runs the benchmark on pre-collected data."""
    print(f"Starting Benchmark Mode: models={models}, samples={samples}, temperature={temperature}")
    benchmark = MapGuesserBenchmark(headless=headless)
    summary = benchmark.run_benchmark(models=models, max_samples=samples, temperature=temperature)
    if summary:
        print("\n--- Benchmark Complete! Summary ---")
        for model, stats in summary.items():
            print(f"Model: {model}")
            print(f"  Success Rate: {stats['success_rate'] * 100:.1f}%")
            print(f"  Avg Distance: {stats['average_distance_km']:.1f} km")


def main():
    parser = argparse.ArgumentParser(description="MapCrunch AI Agent & Benchmark")
    parser.add_argument(
        "--mode",
        choices=["agent", "benchmark"],
        default="agent",
        help="Operation mode.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS_CONFIG.keys()),
        default="gpt-4o",
        help="Model to use.",
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="[Agent] Number of exploration steps."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to process for the selected mode.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS_CONFIG.keys()),
        help="[Benchmark] Models to benchmark.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature parameter for LLM sampling (0.0 = deterministic, higher = more random). Default: 0.0",
    )

    args = parser.parse_args()

    if args.mode == "agent":
        agent_mode(
            model_name=args.model,
            steps=args.steps,
            headless=args.headless,
            samples=args.samples,
            temperature=args.temperature,
        )
    elif args.mode == "benchmark":
        benchmark_mode(
            models=args.models or [args.model],
            samples=args.samples,
            headless=args.headless,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
