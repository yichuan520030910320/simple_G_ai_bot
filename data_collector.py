import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import uuid

from mapcrunch_controller import MapCrunchController
from config import DATA_PATHS, BENCHMARK_CONFIG


class DataCollector:
    """Collect MapCrunch location identifiers and coordinates (no screenshots needed)"""
    
    def __init__(self, headless: bool = False):
        self.controller = MapCrunchController(headless=headless)
        self.data = []
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for data storage"""
        for path in DATA_PATHS.values():
            if path.endswith('/'):
                Path(path).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def collect_samples(self, num_samples: int = None) -> List[Dict]:
        """Collect specified number of MapCrunch locations with coordinates"""
        if num_samples is None:
            num_samples = BENCHMARK_CONFIG['data_collection_samples']
        
        print(f"🚀 Starting location data collection for {num_samples} samples...")
        print("📍 Collecting location identifiers and coordinates (no screenshots)")
        
        # Setup clean environment
        self.controller.setup_clean_environment()
        
        successful_samples = 0
        failed_samples = 0
        
        for i in range(num_samples):
            try:
                print(f"\n📍 Collecting location {i+1}/{num_samples}")
                
                # Get new random location
                if not self.controller.click_go_button():
                    print("❌ Failed to get new location")
                    failed_samples += 1
                    continue
                
                # Brief wait for page to load
                time.sleep(1)
                
                # Collect location data (no screenshot needed)
                location_data = self.collect_single_location()
                
                if location_data:
                    self.data.append(location_data)
                    successful_samples += 1
                    lat, lng = location_data.get('lat'), location_data.get('lng')
                    if lat and lng:
                        print(f"✅ Location {i+1}: {lat:.4f}, {lng:.4f}")
                    else:
                        print(f"✅ Location {i+1}: {location_data.get('address', 'Unknown')}")
                else:
                    failed_samples += 1
                    print(f"❌ Location {i+1} failed")
                
                # Brief pause between samples
                time.sleep(0.3)
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Collection stopped by user after {successful_samples} samples")
                break
            except Exception as e:
                print(f"❌ Error collecting location {i+1}: {e}")
                failed_samples += 1
                continue
        
        print(f"\n📊 Collection Summary:")
        print(f"✅ Successful: {successful_samples}")
        print(f"❌ Failed: {failed_samples}")
        print(f"📈 Success rate: {successful_samples/(successful_samples+failed_samples)*100:.1f}%")
        
        # Save collected data
        self.save_data()
        
        return self.data
    
    def collect_single_location(self) -> Optional[Dict]:
        """Collect a single location identifier and coordinates"""
        try:
            sample_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Get current coordinates and location identifiers from page
            coordinates = self.controller.get_current_coordinates()
            if not coordinates:
                print("⚠️  Could not extract coordinates from page")
                return None
            
            # Get current URL and page state for later reproduction
            current_url = self.controller.driver.current_url
            
            # Extract MapCrunch-specific identifiers
            location_identifiers = self.controller.driver.execute_script("""
                try {
                    return {
                        initString: typeof initString !== 'undefined' ? initString : null,
                        initPanoId: typeof initPanoId !== 'undefined' ? initPanoId : null,
                        permLink: typeof gPermLink !== 'undefined' ? gPermLink : null,
                        locationString: typeof initLocationString !== 'undefined' ? initLocationString : null,
                        url: window.location.href
                    };
                } catch (error) {
                    return {url: window.location.href};
                }
            """)
            
            # Validate we got useful data
            if coordinates.get('lat') is None or coordinates.get('lng') is None:
                if coordinates.get('address'):
                    print(f"⚠️  Only got address: {coordinates['address']}")
                    # Still save address-only samples for potential geocoding later
                else:
                    print("⚠️  No valid coordinates or address found")
                    return None
            
            # Create location data record (no screenshot!)
            location_data = {
                'id': sample_id,
                'timestamp': timestamp,
                'coordinates': coordinates,
                'lat': coordinates.get('lat'),
                'lng': coordinates.get('lng'),
                'address': coordinates.get('address'),
                'source': coordinates.get('source', 'unknown'),
                'identifiers': location_identifiers,
                'url': current_url,
                'init_string': location_identifiers.get('initString'),
                'pano_id': location_identifiers.get('initPanoId'),
                'perm_link': location_identifiers.get('permLink')
            }
            
            return location_data
            
        except Exception as e:
            print(f"❌ Error in collect_single_location: {e}")
            return None
    
    def save_data(self):
        """Save collected location data to JSON file"""
        try:
            output_data = {
                'metadata': {
                    'collection_date': datetime.now().isoformat(),
                    'total_samples': len(self.data),
                    'version': '2.0',
                    'description': 'MapCrunch location identifiers and coordinates for benchmark testing'
                },
                'samples': self.data
            }
            
            with open(DATA_PATHS['golden_labels'], 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Location data saved to {DATA_PATHS['golden_labels']}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def load_existing_data(self) -> List[Dict]:
        """Load existing location data"""
        try:
            if os.path.exists(DATA_PATHS['golden_labels']):
                with open(DATA_PATHS['golden_labels'], 'r') as f:
                    data = json.load(f)
                return data.get('samples', [])
            else:
                return []
        except Exception as e:
            print(f"❌ Error loading existing data: {e}")
            return []
    
    def validate_sample(self, sample: Dict) -> bool:
        """Validate that a sample has required fields"""
        required_fields = ['id', 'coordinates']
        
        # Check required fields
        if not all(field in sample for field in required_fields):
            return False
        
        # Check if coordinates are valid
        coords = sample['coordinates']
        if coords.get('lat') is None or coords.get('lng') is None:
            if coords.get('address') is None:
                return False
        
        return True
    
    def clean_invalid_samples(self):
        """Remove invalid samples from dataset"""
        existing_data = self.load_existing_data()
        valid_samples = [sample for sample in existing_data if self.validate_sample(sample)]
        
        print(f"🧹 Cleaned dataset: {len(existing_data)} -> {len(valid_samples)} samples")
        
        if len(valid_samples) != len(existing_data):
            # Save cleaned data
            self.data = valid_samples
            self.save_data()
    
    def close(self):
        """Clean up resources"""
        self.controller.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Main function for data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect MapCrunch location data for benchmark')
    parser.add_argument('--samples', type=int, default=50, help='Number of locations to collect')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--clean', action='store_true', help='Clean invalid samples from existing data')
    
    args = parser.parse_args()
    
    if args.clean:
        print("🧹 Cleaning existing dataset...")
        with DataCollector(headless=True) as collector:
            collector.clean_invalid_samples()
        return
    
    # Collect new location data
    with DataCollector(headless=args.headless) as collector:
        data = collector.collect_samples(args.samples)
        print(f"\n🎉 Collection complete! Collected {len(data)} location samples.")
        print(f"📊 Ready for benchmark testing with these locations.")


if __name__ == "__main__":
    main()