# benchmark.py (Final Version)

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import math

from geo_bot import GeoBot
from config import DATA_PATHS, MODELS_CONFIG, SUCCESS_THRESHOLD_KM


class MapGuesserBenchmark:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.golden_labels = self.load_golden_labels()
        print(f"📊 Loaded {len(self.golden_labels)} golden label samples")

    def load_golden_labels(self) -> List[Dict]:
        try:
            with open(DATA_PATHS["golden_labels"], "r") as f:
                return json.load(f).get("samples", [])
        except Exception:
            return []

    def get_model_class(self, model_name: str):
        config = MODELS_CONFIG.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")
        class_name, model_class_name = config["class"], config["model_name"]
        if class_name == "ChatOpenAI":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI, model_class_name
        if class_name == "ChatAnthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic, model_class_name
        if class_name == "ChatGoogleGenerativeAI":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI, model_class_name
        raise ValueError(f"Unknown model class: {class_name}")

    def calculate_distance(
        self, true_coords: Dict, predicted_coords: Optional[Tuple[float, float]]
    ) -> Optional[float]:
        """Calculates distance between true (lat,lon) and predicted (lat,lon)."""
        if not predicted_coords:
            return None
        try:
            true_lat, true_lng = true_coords["lat"], true_coords["lng"]
            pred_lat, pred_lng = predicted_coords

            R = 6371
            lat1, lon1, lat2, lon2 = map(
                math.radians, [true_lat, true_lng, pred_lat, pred_lng]
            )
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c
        except (TypeError, KeyError, IndexError) as e:
            print(f"Error in distance calculation: {e}")
            return None

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        if not self.golden_labels:
            raise ValueError("No golden labels available.")

        models_to_test = models or list(MODELS_CONFIG.keys())
        test_samples = self.golden_labels[:max_samples]

        print(f"🚀 Starting LIVE benchmark:")
        print(f"   Models: {models_to_test}")
        print(f"   Samples: {len(test_samples)}")

        all_results = []
        for model_name in models_to_test:
            print(f"\n🤖 Testing model: {model_name}")
            model_class, model_class_name = self.get_model_class(model_name)

            try:
                with GeoBot(
                    model=model_class,
                    model_name=model_class_name,
                    use_selenium=True,
                    headless=self.headless,
                ) as bot:
                    for i, sample in enumerate(test_samples):
                        print(f"   📍 Sample {i + 1}/{len(test_samples)}")
                        try:
                            result = self.run_single_test_with_bot(bot, sample)
                            all_results.append(result)

                            status = (
                                "✅ Success" if result.get("success") else "❌ Failed"
                            )
                            distance = result.get("distance_km")
                            dist_str = (
                                f"{distance:.1f} km" if distance is not None else "N/A"
                            )
                            print(f"   {status} (Distance: {dist_str})")

                        except KeyboardInterrupt:
                            print("\n⏹️  Benchmark inner loop interrupted.")
                            raise
                        except Exception as e:
                            print(f"   ❌ Test failed with unhandled exception: {e}")
                            all_results.append(
                                {
                                    "model": model_name,
                                    "sample_id": sample["id"],
                                    "success": False,
                                    "error": str(e),
                                }
                            )

            except KeyboardInterrupt:
                print("\n⏹️  Benchmark outer loop interrupted.")
                break

        self.save_results(all_results)
        return self.generate_summary(all_results)

    def run_single_test_with_bot(self, bot: GeoBot, location_data: Dict) -> Dict:
        start_time = time.time()

        assert bot.controller is not None
        if not bot.controller.load_location_from_data(location_data):
            return {
                "success": False,
                "error": "Failed to load location",
                "model": bot.model_name,
                "sample_id": location_data["id"],
            }

        screenshot = bot.take_screenshot()
        if not screenshot:
            return {
                "success": False,
                "error": "Failed to take screenshot",
                "model": bot.model_name,
                "sample_id": location_data["id"],
            }

        predicted_lat_lon = bot.analyze_image(screenshot)
        inference_time = time.time() - start_time

        true_coords = location_data["coordinates"]
        distance_km = self.calculate_distance(true_coords, predicted_lat_lon)

        is_success = distance_km is not None and distance_km <= SUCCESS_THRESHOLD_KM

        return {
            "sample_id": location_data["id"],
            "model": bot.model_name,
            "true_coordinates": true_coords,
            "predicted_coordinates": predicted_lat_lon,
            "distance_km": distance_km,
            "inference_time": inference_time,
            "success": is_success,
        }

    def save_results(self, results: List[Dict]):
        if not results:
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(DATA_PATHS["results"])
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / f"benchmark_results_{timestamp}.json"
            output_data = {
                "metadata": {"timestamp": datetime.now().isoformat()},
                "results": results,
            }
            with open(results_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"💾 Results saved to {results_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def generate_summary(self, results: List[Dict]) -> Dict:
        summary = {}
        by_model = {}
        for r in results:
            model = r.get("model", "unknown")
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)

        for model, model_results in by_model.items():
            successful_runs = [r for r in model_results if r.get("success")]
            distances = [
                r["distance_km"]
                for r in model_results
                if r.get("distance_km") is not None
            ]

            if not model_results:
                continue

            summary[model] = {
                "success_rate": len(successful_runs) / len(model_results)
                if model_results
                else 0,
                "average_distance_km": sum(distances) / len(distances)
                if distances
                else None,
                "median_distance_km": sorted(distances)[len(distances) // 2]
                if distances
                else None,
                "min_distance_km": min(distances) if distances else None,
                "max_distance_km": max(distances) if distances else None,
            }
        return summary
