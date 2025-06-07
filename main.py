#!/usr/bin/env python3
"""
Main entry point for MapCrunch geo-location testing

Usage:
    python main.py --mode data --samples 50 --urban --no-indoor   # Collect filtered data
    python main.py --mode benchmark --models gpt-4o claude-3.5-sonnet  # Run benchmark
    python main.py --mode interactive --model gpt-4o  # Interactive testing
"""

import argparse
import os
from time import sleep
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from geo_bot import GeoBot
from data_collector import DataCollector
from benchmark import MapGuesserBenchmark
from config import MODELS_CONFIG, SUCCESS_THRESHOLD_KM


def interactive_mode(model_name: str = "gpt-4o", turns: int = 5, plot: bool = False):
    """Interactive mode - play turns manually like the original"""
    print(f"🎮 Starting interactive mode with {model_name}")

    # Get model class
    config = MODELS_CONFIG.get(model_name)
    if not config:
        print(f"❌ Unknown model: {model_name}")
        return

    model_class_name = config["class"]
    model_class = globals()[model_class_name]
    model_instance = config["model_name"]

    # Create bot with Selenium integration
    with GeoBot(model=model_class, model_name=model_instance, use_selenium=True) as bot:
        # Setup clean environment
        if bot.controller:
            bot.controller.setup_clean_environment()

        for turn in range(turns):
            print(f"\n{'=' * 50}")
            print(f"🎯 Turn {turn + 1}/{turns}")
            print(f"{'=' * 50}")

            try:
                # Get new location (click Go button)
                if bot.controller:
                    if not bot.controller.click_go_button():
                        print("❌ Failed to get new location")
                        continue
                else:
                    print("⚠️  Manual mode: Please click Go button and press Enter")
                    input()

                # Take screenshot and analyze
                screenshot = bot.take_screenshot()
                location = bot.analyze_image(screenshot)

                if location is not None:
                    bot.select_map_location(*location, plot=plot)
                    print("✅ Location selected successfully")
                else:
                    print("❌ Could not determine location")
                    # Select a default location
                    bot.select_map_location(
                        x=bot.map_x + bot.map_w // 2,
                        y=bot.map_y + bot.map_h // 2,
                        plot=plot,
                    )

                # Brief pause between turns
                sleep(2)

            except KeyboardInterrupt:
                print(f"\n⏹️  Game stopped by user after {turn + 1} turns")
                break
            except Exception as e:
                print(f"❌ Error in turn {turn + 1}: {e}")
                continue


def data_collection_mode(
    samples: int = 50, headless: bool = False, options: Dict = None
):
    """Data collection mode"""
    print(f"📊 Starting data collection mode - {samples} samples")

    if options:
        print(f"🔧 Using custom options: {options}")

    with DataCollector(headless=headless, options=options) as collector:
        data = collector.collect_samples(samples)
        print(f"✅ Collected {len(data)} samples successfully")


def benchmark_mode(
    models: list = None, samples: int = 10, live: bool = False, headless: bool = False
):
    """Benchmark mode"""
    if models is None:
        models = ["gpt-4o"]  # Default model

    print(f"🏁 Starting benchmark mode")
    print(f"   Models: {models}")
    print(f"   Samples per model: {samples}")
    print(f"   Mode: {'live' if live else 'offline'}")

    benchmark = MapGuesserBenchmark(headless=headless)

    try:
        summary = benchmark.run_benchmark(
            models=models, max_samples=samples, use_live_mode=live
        )

        print(f"\n🎉 Benchmark Complete!")

        if summary:
            print(f"\n📊 Results Summary:")
            for model, stats in summary.items():
                print(f"\n🤖 {model}:")
                print(
                    f"   Success Rate (under {SUCCESS_THRESHOLD_KM}km): {stats.get('success_rate', 0) * 100:.1f}%"
                )
                print(f"   📏 Average Distance: {stats['average_distance_km']:.1f} km")
                print(f"   📊 Median Distance: {stats['median_distance_km']:.1f} km")
                print(f"   🎯 Best: {stats['min_distance_km']:.1f} km")
                print(f"   📈 Worst: {stats['max_distance_km']:.1f} km")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="MapCrunch Geo-Location AI Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect training data with filters
  python main.py --mode data --samples 100 --urban --no-indoor
  
  # Collect from specific countries
  python main.py --mode data --samples 50 --countries us gb jp --urban
  
  # Run benchmark on saved data
  python main.py --mode benchmark --models gpt-4o claude-3.5-sonnet --samples 20
  
  # Interactive testing  
  python main.py --mode interactive --model gpt-4o --turns 5 --plot
  
  # Live benchmark (uses MapCrunch website directly)
  python main.py --mode benchmark --live --models gpt-4o
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["interactive", "data", "benchmark"],
        default="interactive",
        help="Operation mode",
    )

    # Interactive mode options
    parser.add_argument(
        "--model",
        choices=list(MODELS_CONFIG.keys()),
        default="gpt-4o",
        help="Model for interactive mode",
    )
    parser.add_argument(
        "--turns", type=int, default=5, help="Number of turns in interactive mode"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate plots of predictions"
    )

    # Data collection options
    parser.add_argument(
        "--samples", type=int, default=50, help="Number of samples to collect/test"
    )
    parser.add_argument(
        "--urban", action="store_true", help="Collect only urban locations"
    )
    parser.add_argument("--no-indoor", action="store_true", help="Exclude indoor views")
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Specific countries to collect from (e.g., us gb jp)",
    )

    # Benchmark options
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS_CONFIG.keys()),
        help="Models to benchmark",
    )
    parser.add_argument(
        "--live", action="store_true", help="Use live MapCrunch website for benchmark"
    )

    # General options
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )

    args = parser.parse_args()

    print(f"🚀 MapCrunch Geo-Location AI Benchmark")
    print(f"   Mode: {args.mode}")

    try:
        if args.mode == "interactive":
            interactive_mode(model_name=args.model, turns=args.turns, plot=args.plot)

        elif args.mode == "data":
            # Configure collection options from args
            from config import MAPCRUNCH_OPTIONS

            options = MAPCRUNCH_OPTIONS.copy()

            if args.urban:
                options["urban_only"] = True
            if args.no_indoor:
                options["exclude_indoor"] = True
            if args.countries:
                options["selected_countries"] = args.countries

            data_collection_mode(
                samples=args.samples, headless=args.headless, options=options
            )

        elif args.mode == "benchmark":
            benchmark_mode(
                models=args.models,
                samples=args.samples,
                live=args.live,
                headless=args.headless,
            )

    except KeyboardInterrupt:
        print(f"\n⏹️  Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
