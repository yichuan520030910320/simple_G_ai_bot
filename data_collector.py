# data_collector.py (Updated for Named Datasets)

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
    get_data_paths,
    BENCHMARK_CONFIG,
    DATA_COLLECTION_CONFIG,
    MAPCRUNCH_OPTIONS,
)


class DataCollector:
    def __init__(
        self,
        dataset_name: str = "default",
        headless: bool = False,
        options: Optional[Dict] = None,
    ):
        self.dataset_name = dataset_name
        self.data_paths = get_data_paths(dataset_name)
        self.controller = MapCrunchController(headless=headless)
        self.data = []
        self.options = options or MAPCRUNCH_OPTIONS
        self.setup_directories()

    def setup_directories(self):
        for path in self.data_paths.values():
            if path.endswith("/"):
                Path(path).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

    def collect_samples(
        self, num_samples: Optional[int] = None, **kwargs
    ) -> List[Dict]:
        num_samples = num_samples or BENCHMARK_CONFIG["data_collection_samples"]
        print(
            f"🚀 Collecting {num_samples} samples for dataset '{self.dataset_name}'..."
        )

        successful_samples = 0
        while successful_samples < num_samples:
            print(f"\n📍 Collecting location {successful_samples + 1}/{num_samples}")
            if not self.controller.click_go_button():
                print("❌ Failed to get new location, retrying...")
                time.sleep(2)
                continue

            location_data = self.collect_single_location()
            if location_data:
                self.data.append(location_data)
                successful_samples += 1
                lat, lng = location_data.get("lat"), location_data.get("lng")
                print(
                    f"✅ Location {successful_samples}: {location_data.get('address', 'N/A')} ({lat:.4f}, {lng:.4f})"
                )
            else:
                print("❌ Location collection failed")

        self.save_data()
        return self.data

    def collect_single_location(self) -> Optional[Dict]:
        """Collects a single location with simplified data collection."""
        try:
            # Get coordinates
            coords = self.controller.driver.execute_script(
                "return { lat: window.panorama.getPosition().lat(), lng: window.panorama.getPosition().lng() };"
            )
            if not coords:
                raise ValueError("Could not get coordinates.")

            # Get POV data directly from panorama
            pov_data = self.controller.driver.execute_script("""
                return {
                    heading: window.panorama.getPov().heading,
                    pitch: window.panorama.getPov().pitch,
                    zoom: window.panorama.getZoom(),
                    panoId: window.panorama.getPano()
                };
            """)

            if not pov_data:
                raise ValueError("Could not get POV data.")

            # Get address (simplified)
            address = "Unknown"
            try:
                address = self.controller.get_current_address() or "Unknown"
            except:
                pass  # Address is optional

            lat = coords.get("lat")
            lng = coords.get("lng")

            # Simplified URL slug construction
            def round_num(n, d):
                return f"{n:.{d}f}"

            zoom_for_slug = max(0, round(pov_data.get("zoom", 1.0)) - 1)
            url_slug = (
                f"{round_num(lat, 6)}_"
                f"{round_num(lng, 6)}_"
                f"{round_num(pov_data.get('heading', 0), 2)}_"
                f"{round_num(pov_data.get('pitch', 0) * -1, 2)}_"
                f"{zoom_for_slug}"
            )

            sample_id = str(uuid.uuid4())
            location_data = {
                "id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "lat": lat,
                "lng": lng,
                "address": address,
                "pano_id": pov_data.get("panoId"),
                "pov": {
                    "heading": pov_data.get("heading", 0),
                    "pitch": pov_data.get("pitch", 0),
                    "zoom": pov_data.get("zoom", 1.0),
                },
                "url_slug": url_slug,
            }

            # Try to save thumbnail (optional)
            thumbnail_path = self.save_thumbnail(sample_id)
            if thumbnail_path:
                location_data["thumbnail_path"] = thumbnail_path

            return location_data

        except Exception as e:
            print(f"❌ Error in collect_single_location: {e}")
            return None

    def save_thumbnail(self, sample_id: str) -> Optional[str]:
        try:
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                print(
                    f"⚠️  Could not take screenshot for {sample_id} (this is OK in headless mode)"
                )
                return None

            image = Image.open(BytesIO(screenshot_bytes))
            thumbnail_size = DATA_COLLECTION_CONFIG.get("thumbnail_size", (320, 240))
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            thumbnail_filename = f"{sample_id}.jpg"
            thumbnail_path = os.path.join(
                self.data_paths["thumbnails"], thumbnail_filename
            )

            if image.mode in ("RGBA", "LA"):
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                image = rgb_image

            image.save(thumbnail_path, "JPEG", quality=85)
            print(f"✅ Saved thumbnail for {sample_id}")
            return thumbnail_filename
        except Exception as e:
            print(f"⚠️  Could not save thumbnail for {sample_id}: {e}")
            return None

    def save_data(self):
        try:
            output_data = {
                "metadata": {
                    "dataset_name": self.dataset_name,
                    "collection_date": datetime.now().isoformat(),
                    "collection_options": self.options,
                },
                "samples": self.data,
            }
            with open(self.data_paths["golden_labels"], "w") as f:
                json.dump(output_data, f, indent=2)
            print(
                f"\n💾 Dataset '{self.dataset_name}' saved to {self.data_paths['golden_labels']}"
            )
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def close(self):
        self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
