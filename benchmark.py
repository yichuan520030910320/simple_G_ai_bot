import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import math
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from geo_bot import GeoBot
from mapcrunch_controller import MapCrunchController
from config import DATA_PATHS, MODELS_CONFIG, BENCHMARK_CONFIG


class MapGuesserBenchmark:
    """Benchmark system for testing geo-location models"""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.results = []
        self.golden_labels = self.load_golden_labels()
        print(f"📊 Loaded {len(self.golden_labels)} golden label samples")
    
    def load_golden_labels(self) -> List[Dict]:
        """Load golden labels from data collection"""
        try:
            if os.path.exists(DATA_PATHS['golden_labels']):
                with open(DATA_PATHS['golden_labels'], 'r') as f:
                    data = json.load(f)
                return data.get('samples', [])
            else:
                print(f"❌ Golden labels file not found: {DATA_PATHS['golden_labels']}")
                print("💡 Run data_collector.py first to collect training data")
                return []
        except Exception as e:
            print(f"❌ Error loading golden labels: {e}")
            return []
    
    def get_model_class(self, model_name: str):
        """Get model class from configuration"""
        config = MODELS_CONFIG.get(model_name)
        if not config:
            raise ValueError(f"Unknown model: {model_name}")
        
        class_name = config['class']
        model_class_name = config['model_name']
        
        if class_name == 'ChatOpenAI':
            return ChatOpenAI, model_class_name
        elif class_name == 'ChatAnthropic':
            return ChatAnthropic, model_class_name
        elif class_name == 'ChatGoogleGenerativeAI':
            return ChatGoogleGenerativeAI, model_class_name
        else:
            raise ValueError(f"Unknown model class: {class_name}")
    
    def calculate_distance(self, true_coords: Dict, predicted_coords: Tuple[float, float]) -> Optional[float]:
        """Calculate distance between true and predicted coordinates in kilometers"""
        try:
            # Extract true coordinates
            true_lat = true_coords.get('lat')
            true_lng = true_coords.get('lng')
            
            if true_lat is None or true_lng is None:
                print("⚠️  No valid coordinates in golden label")
                return None
            
            if predicted_coords is None:
                return None
            
            # Convert screen coordinates back to lat/lon if needed
            if len(predicted_coords) == 2 and predicted_coords[0] > 180:
                # These look like screen coordinates, not lat/lon
                print("⚠️  Predicted coordinates appear to be screen pixels, not lat/lon")
                return None
            
            pred_lat, pred_lng = predicted_coords
            
            # Haversine formula
            R = 6371  # Earth's radius in kilometers
            
            lat1, lon1 = math.radians(true_lat), math.radians(true_lng)
            lat2, lon2 = math.radians(pred_lat), math.radians(pred_lng)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            distance_km = R * c
            return distance_km
            
        except Exception as e:
            print(f"❌ Error calculating distance: {e}")
            return None
    
    def run_single_test(self, model_name: str, location_data: Dict) -> Dict:
        """Run a single test with a model and location"""
        try:
            print(f"\n🧪 Testing {model_name} on location {location_data['id'][:8]}...")
            
            # Get model
            model_class, model_class_name = self.get_model_class(model_name)
            
            # Always use Selenium for benchmark (need to load locations and take screenshots)
            result = self.run_selenium_test(model_class, model_class_name, location_data)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in single test: {e}")
            return {
                'sample_id': location_data['id'],
                'model': model_name,
                'error': str(e),
                'success': False
            }
    
    def run_selenium_test(self, model_class, model_class_name: str, location_data: Dict) -> Dict:
        """Run test using Selenium to recreate location and test model"""
        
        # Create bot with Selenium
        with GeoBot(model=model_class, model_name=model_class_name, use_selenium=True) as bot:
            
            start_time = time.time()
            
            # Load the specific location from data
            success = bot.controller.load_location_from_data(location_data)
            if not success:
                raise Exception("Could not load location from data")
            
            # Setup clean environment for testing
            bot.controller.setup_clean_environment()
            
            # Take screenshot of Street View
            screenshot = bot.take_screenshot()
            if not screenshot:
                raise Exception("Could not take screenshot")
            
            # Analyze with LLM
            prediction = bot.analyze_image(screenshot)
            
            inference_time = time.time() - start_time
            
            # Extract true coordinates from original data
            true_coords = location_data['coordinates']
            
            # Calculate distance
            distance_km = self.calculate_distance(true_coords, prediction)
            
            return {
                'sample_id': location_data['id'],
                'model': model_class_name,
                'true_coordinates': true_coords,
                'predicted_coordinates': prediction,
                'distance_km': distance_km,
                'inference_time': inference_time,
                'success': prediction is not None,
                'mode': 'selenium'
            }
    
    def run_benchmark(self, models: List[str] = None, max_samples: int = None, 
                     use_live_mode: bool = False) -> Dict:
        """Run complete benchmark across models and samples"""
        
        if not self.golden_labels:
            raise ValueError("No golden labels available. Run data collection first.")
        
        if models is None:
            models = list(MODELS_CONFIG.keys())
        
        if max_samples is None:
            max_samples = BENCHMARK_CONFIG['rounds_per_model']
        
        # Limit samples
        test_samples = self.golden_labels[:max_samples]
        
        print(f"🚀 Starting benchmark:")
        print(f"   Models: {models}")
        print(f"   Samples: {len(test_samples)}")
        print(f"   Mode: {'live' if use_live_mode else 'offline'}")
        
        all_results = []
        
        for model_name in models:
            print(f"\n🤖 Testing model: {model_name}")
            
            model_results = []
            successful_tests = 0
            
            for i, sample in enumerate(test_samples):
                try:
                    print(f"   📍 Sample {i+1}/{len(test_samples)}")
                    
                    result = self.run_single_test(model_name, sample, use_live_mode)
                    
                    if result.get('success', False):
                        successful_tests += 1
                        distance = result.get('distance_km', float('inf'))
                        print(f"   ✅ Distance: {distance:.1f} km")
                    else:
                        print(f"   ❌ Failed")
                    
                    model_results.append(result)
                    all_results.append(result)
                    
                except KeyboardInterrupt:
                    print(f"\n⏹️  Benchmark interrupted by user")
                    break
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
            
            # Model summary
            if model_results:
                distances = [r['distance_km'] for r in model_results if r.get('distance_km') is not None]
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    median_distance = sorted(distances)[len(distances)//2]
                    print(f"\n📊 {model_name} Summary:")
                    print(f"   Success rate: {successful_tests}/{len(test_samples)} ({successful_tests/len(test_samples)*100:.1f}%)")
                    print(f"   Average distance: {avg_distance:.1f} km")
                    print(f"   Median distance: {median_distance:.1f} km")
        
        # Save results
        self.save_results(all_results)
        
        # Generate summary
        summary = self.generate_summary(all_results)
        
        return summary
    
    def save_results(self, results: List[Dict]):
        """Save benchmark results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(DATA_PATHS['results'], f"benchmark_results_{timestamp}.json")
            
            output_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_tests': len(results),
                    'models_tested': list(set(r['model'] for r in results if 'model' in r)),
                    'version': '1.0'
                },
                'results': results
            }
            
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Results saved to {results_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from results"""
        
        summary = {}
        
        # Group by model
        by_model = {}
        for result in results:
            model = result.get('model', 'unknown')
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        # Calculate statistics for each model
        for model, model_results in by_model.items():
            successful = [r for r in model_results if r.get('success', False)]
            distances = [r['distance_km'] for r in successful if r.get('distance_km') is not None]
            
            if distances:
                summary[model] = {
                    'total_tests': len(model_results),
                    'successful_tests': len(successful),
                    'success_rate': len(successful) / len(model_results),
                    'average_distance_km': sum(distances) / len(distances),
                    'median_distance_km': sorted(distances)[len(distances)//2],
                    'min_distance_km': min(distances),
                    'max_distance_km': max(distances),
                    'distances': distances
                }
        
        return summary


def main():
    """Main function for running benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MapCrunch geo-location benchmark')
    parser.add_argument('--models', nargs='+', choices=list(MODELS_CONFIG.keys()), 
                       help='Models to test')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of samples to test per model')
    parser.add_argument('--live', action='store_true',
                       help='Use live MapCrunch website instead of saved screenshots')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode (live mode only)')
    
    args = parser.parse_args()
    
    benchmark = MapGuesserBenchmark(headless=args.headless)
    
    try:
        summary = benchmark.run_benchmark(
            models=args.models,
            max_samples=args.samples,
            use_live_mode=args.live
        )
        
        print(f"\n🎉 Benchmark Complete!")
        print(f"\n📊 Final Summary:")
        for model, stats in summary.items():
            print(f"\n{model}:")
            print(f"   Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"   Average Distance: {stats['average_distance_km']:.1f} km")
            print(f"   Median Distance: {stats['median_distance_km']:.1f} km")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark error: {e}")


if __name__ == "__main__":
    main()