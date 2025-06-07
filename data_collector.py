import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import uuid
from PIL import Image
from io import BytesIO

from mapcrunch_controller import MapCrunchController
from config import (
    DATA_PATHS,
    BENCHMARK_CONFIG,
    DATA_COLLECTION_CONFIG,
    MAPCRUNCH_OPTIONS,
)


class DataCollector:
    """Collect MapCrunch location identifiers, coordinates, and thumbnails"""

    def __init__(self, headless: bool = False, options: Optional[Dict] = None):
        self.controller = MapCrunchController(headless=headless)
        self.data = []
        self.options = options or MAPCRUNCH_OPTIONS
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for data storage"""
        for path in DATA_PATHS.values():
            if path.endswith("/"):
                Path(path).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

    def collect_samples(
        self, num_samples: Optional[int] = None, filter_indoor: Optional[bool] = None
    ) -> List[Dict]:
        """Collect specified number of MapCrunch locations with coordinates and thumbnails"""
        if num_samples is None:
            num_samples = BENCHMARK_CONFIG["data_collection_samples"]

        # Override indoor filter if specified
        if filter_indoor is not None:
            self.options["exclude_indoor"] = filter_indoor

        print(f"🚀 Starting location data collection for {num_samples} samples...")
        print(
            f"📍 Options: Urban={self.options.get('urban_only', False)}, Exclude Indoor={self.options.get('exclude_indoor', True)}"
        )

        # Setup MapCrunch options
        if not self.controller.setup_collection_options(self.options):
            print("⚠️  Could not configure all options, continuing anyway...")

        # Setup clean environment for stealth mode if needed
        if self.options.get("stealth_mode", True):
            self.controller.setup_clean_environment()

        successful_samples = 0
        failed_samples = 0
        consecutive_failures = 0

        while successful_samples < num_samples:
            try:
                print(
                    f"\n📍 Collecting location {successful_samples + 1}/{num_samples}"
                )

                # Get new random location
                if not self.controller.click_go_button():
                    print("❌ Failed to get new location")
                    failed_samples += 1
                    consecutive_failures += 1
                    if consecutive_failures > 5:
                        print("❌ Too many consecutive failures, stopping")
                        break
                    continue

                # Wait for page to load
                time.sleep(DATA_COLLECTION_CONFIG.get("wait_after_go", 5))

                # Collect location data with retries
                location_data = None
                retries = (
                    DATA_COLLECTION_CONFIG.get("max_retries", 3)
                    if DATA_COLLECTION_CONFIG.get("retry_on_failure", True)
                    else 1
                )

                for retry in range(retries):
                    location_data = self.collect_single_location()
                    if location_data:
                        break
                    if retry < retries - 1:
                        print(f"   ⚠️  Retry {retry + 1}/{retries - 1}")
                        time.sleep(1)

                if location_data:
                    self.data.append(location_data)
                    successful_samples += 1
                    consecutive_failures = 0

                    # Display collected info
                    address = location_data.get("address", "Unknown")
                    lat, lng = location_data.get("lat"), location_data.get("lng")
                    if lat and lng:
                        print(
                            f"✅ Location {successful_samples}: {address} ({lat:.4f}, {lng:.4f})"
                        )
                    else:
                        print(f"✅ Location {successful_samples}: {address}")

                    if location_data.get("thumbnail_path"):
                        print(
                            f"   📸 Thumbnail saved: {location_data['thumbnail_path']}"
                        )
                else:
                    failed_samples += 1
                    consecutive_failures += 1
                    print("❌ Location collection failed")

                # Brief pause between samples
                time.sleep(0.5)

            except KeyboardInterrupt:
                print(
                    f"\n⏹️  Collection stopped by user after {successful_samples} samples"
                )
                break
            except Exception as e:
                print(f"❌ Error collecting location: {e}")
                failed_samples += 1
                consecutive_failures += 1
                continue

        print("\n📊 Collection Summary:")
        print(f"✅ Successful: {successful_samples}")
        print(f"❌ Failed: {failed_samples}")
        print(
            f"📈 Success rate: {successful_samples / (successful_samples + failed_samples) * 100:.1f}%"
        )

        # Save collected data
        self.save_data()

        return self.data

    def collect_single_location(self) -> Optional[Dict]:
        """Collect a single location with all metadata"""
        try:
            sample_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            assert self.controller.driver is not None

            # 1. 获取实时坐标 (这个方法依然正确)
            current_coords = self.controller.driver.execute_script(
                "if (window.panorama) { return { lat: window.panorama.getPosition().lat(), lng: window.panorama.getPosition().lng() }; } else { return null; }"
            )
            if not current_coords or current_coords.get("lat") is None:
                return None

            # **2. 新增: 获取实时的链接和Pano ID**
            live_identifiers = self.controller.get_live_location_identifiers()
            if not live_identifiers or "error" in live_identifiers:
                print(
                    f"⚠️ Could not get live identifiers: {live_identifiers.get('error')}"
                )
                return None

            # 3. 获取地址
            address = self.controller.get_current_address()

            # 4. 创建数据记录
            location_data = {
                "id": sample_id,
                "timestamp": timestamp,
                "coordinates": current_coords,
                "lat": current_coords.get("lat"),
                "lng": current_coords.get("lng"),
                "address": address or "Unknown",
                "source": "panorama_object",
                # **使用新的实时标识符**
                "url": live_identifiers.get("permLink"),
                "perm_link": live_identifiers.get("permLink"),
                "pano_id": live_identifiers.get("panoId"),
                "url_slug": live_identifiers.get("urlString"),  # 新增，更可靠
                "collection_options": self.options.copy(),
            }

            # ... (后续保存缩略图的代码不变) ...
            if DATA_COLLECTION_CONFIG.get("save_thumbnails", True):
                thumbnail_path = self.save_thumbnail(sample_id)
                location_data["thumbnail_path"] = thumbnail_path
                location_data["has_thumbnail"] = bool(thumbnail_path)

            # Save full screenshot if configured (storage intensive)
            if DATA_COLLECTION_CONFIG.get("save_full_screenshots", False):
                screenshot_path = self.save_full_screenshot(sample_id)
                if screenshot_path:
                    location_data["screenshot_path"] = screenshot_path

            return location_data

        except Exception as e:
            print(f"❌ Error in collect_single_location: {e}")
            return None

    def save_thumbnail(self, sample_id: str) -> Optional[str]:
        """Save a thumbnail of the current Street View"""
        try:
            # Take screenshot
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                return None

            # Convert to PIL Image
            image = Image.open(BytesIO(screenshot_bytes))

            # Resize to thumbnail size
            thumbnail_size = DATA_COLLECTION_CONFIG.get("thumbnail_size", (320, 240))
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

            # Save thumbnail
            thumbnail_filename = f"{sample_id}.jpg"
            thumbnail_path = os.path.join(DATA_PATHS["thumbnails"], thumbnail_filename)

            # Convert to RGB if necessary (remove alpha channel)
            if image.mode in ("RGBA", "LA"):
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(
                    image, mask=image.split()[-1] if image.mode == "RGBA" else None
                )
                image = rgb_image

            image.save(thumbnail_path, "JPEG", quality=85, optimize=True)

            return thumbnail_filename

        except Exception as e:
            print(f"⚠️  Error saving thumbnail: {e}")
            return None

    def save_full_screenshot(self, sample_id: str) -> Optional[str]:
        """Save full resolution screenshot (optional, storage intensive)"""
        try:
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                return None

            screenshot_filename = f"{sample_id}.png"
            screenshot_path = os.path.join(
                DATA_PATHS["screenshots"], screenshot_filename
            )

            with open(screenshot_path, "wb") as f:
                f.write(screenshot_bytes)

            return screenshot_filename

        except Exception as e:
            print(f"⚠️  Error saving screenshot: {e}")
            return None

    def save_data(self):
        """Save collected location data to JSON file"""
        try:
            # Calculate statistics
            stats = {
                "total_samples": len(self.data),
                "with_coordinates": sum(
                    1 for d in self.data if d.get("lat") is not None
                ),
                "with_address": sum(
                    1
                    for d in self.data
                    if d.get("address") and d["address"] != "Unknown"
                ),
                "with_thumbnails": sum(
                    1 for d in self.data if d.get("has_thumbnail", False)
                ),
                "unique_countries": len(
                    set(
                        d.get("address", "").split(", ")[-1]
                        for d in self.data
                        if d.get("address")
                    )
                ),
            }

            output_data = {
                "metadata": {
                    "collection_date": datetime.now().isoformat(),
                    "total_samples": len(self.data),
                    "statistics": stats,
                    "collection_options": self.options,
                    "version": "3.0",
                    "description": "MapCrunch location data with thumbnails and metadata",
                },
                "samples": self.data,
            }

            with open(DATA_PATHS["golden_labels"], "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\n💾 Location data saved to {DATA_PATHS['golden_labels']}")
            print("📊 Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def load_existing_data(self) -> List[Dict]:
        """Load existing location data"""
        try:
            if os.path.exists(DATA_PATHS["golden_labels"]):
                with open(DATA_PATHS["golden_labels"], "r") as f:
                    data = json.load(f)
                return data.get("samples", [])
            else:
                return []
        except Exception as e:
            print(f"❌ Error loading existing data: {e}")
            return []

    def validate_sample(self, sample: Dict) -> bool:
        """Validate that a sample has required fields"""
        required_fields = ["id", "coordinates"]

        # Check required fields
        if not all(field in sample for field in required_fields):
            return False

        # Check if coordinates are valid
        coords = sample["coordinates"]
        if coords.get("lat") is None or coords.get("lng") is None:
            if coords.get("address") is None:
                return False

        return True

    def clean_invalid_samples(self):
        """Remove invalid samples from dataset"""
        existing_data = self.load_existing_data()
        valid_samples = [
            sample for sample in existing_data if self.validate_sample(sample)
        ]

        print(
            f"🧹 Cleaned dataset: {len(existing_data)} -> {len(valid_samples)} samples"
        )

        if len(valid_samples) != len(existing_data):
            # Save cleaned data
            self.data = valid_samples
            self.save_data()

    def filter_samples(self, filter_func=None, country=None, has_coordinates=None):
        """Filter existing samples based on criteria"""
        samples = self.load_existing_data()

        filtered = samples

        # Filter by country
        if country:
            filtered = [
                s for s in filtered if country.lower() in s.get("address", "").lower()
            ]

        # Filter by coordinate availability
        if has_coordinates is not None:
            if has_coordinates:
                filtered = [
                    s
                    for s in filtered
                    if s.get("lat") is not None and s.get("lng") is not None
                ]
            else:
                filtered = [
                    s for s in filtered if s.get("lat") is None or s.get("lng") is None
                ]

        # Apply custom filter
        if filter_func:
            filtered = [s for s in filtered if filter_func(s)]

        print(f"🔍 Filtered: {len(samples)} -> {len(filtered)} samples")
        return filtered

    def export_summary(self, output_file: str = "data_summary.txt"):
        """Export a human-readable summary of collected data"""
        samples = self.load_existing_data()

        with open(output_file, "w") as f:
            f.write("MapCrunch Data Collection Summary\n")
            f.write("=" * 50 + "\n\n")

            for i, sample in enumerate(samples):
                f.write(f"Sample {i + 1}:\n")
                f.write(f"  ID: {sample['id'][:8]}...\n")
                f.write(f"  Address: {sample.get('address', 'Unknown')}\n")
                f.write(
                    f"  Coordinates: {sample.get('lat', 'N/A')}, {sample.get('lng', 'N/A')}\n"
                )
                f.write(
                    f"  Thumbnail: {'Yes' if sample.get('has_thumbnail') else 'No'}\n"
                )
                f.write(f"  Collected: {sample.get('timestamp', 'Unknown')}\n")
                f.write("-" * 30 + "\n")

        print(f"📄 Summary exported to {output_file}")

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

    parser = argparse.ArgumentParser(
        description="Collect MapCrunch location data for benchmark"
    )
    parser.add_argument(
        "--samples", type=int, default=50, help="Number of locations to collect"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean invalid samples from existing data"
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
    parser.add_argument(
        "--export-summary", action="store_true", help="Export summary of collected data"
    )
    parser.add_argument(
        "--filter-country", help="Filter samples by country when exporting"
    )

    args = parser.parse_args()

    if args.clean:
        print("🧹 Cleaning existing dataset...")
        with DataCollector(headless=True) as collector:
            collector.clean_invalid_samples()
        return

    if args.export_summary:
        print("📄 Exporting data summary...")
        with DataCollector(headless=True) as collector:
            if args.filter_country:
                samples = collector.filter_samples(country=args.filter_country)
                collector.data = samples
                collector.export_summary(f"data_summary_{args.filter_country}.txt")
            else:
                collector.export_summary()
        return

    # Configure collection options
    options = MAPCRUNCH_OPTIONS.copy()

    if args.urban:
        options["urban_only"] = True

    if args.no_indoor:
        options["exclude_indoor"] = True

    if args.countries:
        options["selected_countries"] = args.countries

    # Collect new location data
    with DataCollector(headless=args.headless, options=options) as collector:
        data = collector.collect_samples(args.samples)
        print(f"\n🎉 Collection complete! Collected {len(data)} location samples.")
        print("📊 Ready for benchmark testing with these locations.")


if __name__ == "__main__":
    main()
