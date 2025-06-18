# benchmark.py (Updated for Named Datasets)
#
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import math

from geo_bot import GeoBot
from config import get_data_paths, MODELS_CONFIG, SUCCESS_THRESHOLD_KM, get_model_class


class MapGuesserBenchmark:
    def __init__(self, dataset_name: str = "default", headless: bool = False):
        self.dataset_name = dataset_name
        self.data_paths = get_data_paths(dataset_name)
        self.headless = headless
        self.golden_labels = self.load_golden_labels()
        print(
            f"📊 Loaded {len(self.golden_labels)} samples from dataset '{dataset_name}'"
        )

    def load_golden_labels(self) -> List[Dict]:
        try:
            with open(self.data_paths["golden_labels"], "r") as f:
                return json.load(f).get("samples", [])
        except Exception:
            return []

    def calculate_distance(
        self, true_coords: Dict, predicted_coords: Optional[Tuple[float, float]]
    ) -> Optional[float]:
        if not predicted_coords or "lat" not in true_coords or "lng" not in true_coords:
            return None
        try:
            true_lat, true_lng = true_coords["lat"], true_coords["lng"]
            pred_lat, pred_lng = predicted_coords
            R = 6371
            lat1, lon1, lat2, lon2 = map(
                math.radians, [true_lat, true_lng, pred_lat, pred_lng]
            )
            a = (
                math.sin((dlat := lat2 - lat1) / 2) ** 2
                + math.cos(lat1)
                * math.cos(lat2)
                * math.sin((dlon := lon2 - lon1) / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return R * c
        except Exception:
            return None

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict:
        if not self.golden_labels:
            raise ValueError(
                f"No golden labels available in dataset '{self.dataset_name}'."
            )

        models_to_test = models or list(MODELS_CONFIG.keys())
        num_to_test = (
            min(max_samples, len(self.golden_labels))
            if max_samples is not None
            else len(self.golden_labels)
        )
        test_samples = self.golden_labels[:num_to_test]

        print(f"🚀 Starting benchmark on dataset '{self.dataset_name}':")
        print(f"   Models: {models_to_test}")
        print(f"   Samples: {len(test_samples)}")
        print(f"   Temperature: {temperature}")

        all_results = []
        for model_name in models_to_test:
            print(f"\n🤖 Testing model: {model_name}")
            model_config = MODELS_CONFIG[model_name]
            model_class = get_model_class(model_config["class"])
            model_class_name = model_config["model_name"]

            try:
                with GeoBot(
                    model=model_class,
                    model_name=model_class_name,
                    use_selenium=True,
                    headless=self.headless,
                    temperature=temperature,
                ) as bot:
                    for i, sample in enumerate(test_samples):
                        print(
                            "########################################################"
                        )
                        print(f"📍 Sample {i + 1}/{len(test_samples)}")
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
                            print(f"{status} (Distance: {dist_str})")

                        except KeyboardInterrupt:
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
                print("\n⏹️  Benchmark outer loop interrupted by user.")
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

        bot.controller.setup_clean_environment()

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

        true_coords = {"lat": location_data.get("lat"), "lng": location_data.get("lng")}

        true_location = location_data["address"]
        print(f"🔍 True location: {true_location}")
        print(f"🔍 True coords: {true_coords}")
        print(f"🔍 Predicted coords: {predicted_lat_lon}")
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
            results_dir = Path(self.data_paths["results"])
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / f"benchmark_results_{timestamp}.json"
            output_data = {
                "metadata": {
                    "dataset_name": self.dataset_name,
                    "timestamp": datetime.now().isoformat(),
                },
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
