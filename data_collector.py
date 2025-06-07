# data_collector.py (Restored to original format)

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
    def __init__(self, headless: bool = False, options: Optional[Dict] = None):
        self.controller = MapCrunchController(headless=headless)
        self.data = []
        self.options = options or MAPCRUNCH_OPTIONS
        self.setup_directories()

    def setup_directories(self):
        for path in DATA_PATHS.values():
            if path.endswith("/"):
                Path(path).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

    def collect_samples(
        self, num_samples: Optional[int] = None, **kwargs
    ) -> List[Dict]:
        # ... (此函数不变) ...
        num_samples = num_samples or BENCHMARK_CONFIG["data_collection_samples"]
        print(f"🚀 Starting location data collection for {num_samples} samples...")
        self.controller.setup_collection_options(self.options)

        successful_samples = 0
        while successful_samples < num_samples:
            print(f"\n📍 Collecting location {successful_samples + 1}/{num_samples}")
            if not self.controller.click_go_button():
                print("❌ Failed to get new location")
                continue

            location_data = self.collect_single_location()
            if location_data:
                self.data.append(location_data)
                successful_samples += 1
                lat, lng = location_data.get("lat"), location_data.get("lng")
                print(
                    f"✅ Location {successful_samples}: {location_data['address']} ({lat:.4f}, {lng:.4f})"
                )
            else:
                print("❌ Location collection failed")

        self.save_data()
        return self.data

    def collect_single_location(self) -> Optional[Dict]:
        """Collects a single location using the original, verbose data format."""
        try:
            sample_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            # 1. 获取实时坐标
            current_coords = self.controller.driver.execute_script(
                "if (window.panorama) { return { lat: window.panorama.getPosition().lat(), lng: window.panorama.getPosition().lng() }; } else { return null; }"
            )
            if not current_coords or current_coords.get("lat") is None:
                return None

            # 2. 获取实时标识符
            live_identifiers = self.controller.get_live_location_identifiers()
            if not live_identifiers or "error" in live_identifiers:
                return None

            # 3. 获取地址
            address = self.controller.get_current_address()

            # 4. **构建您期望的、未精简的JSON结构**
            location_data = {
                "id": sample_id,
                "timestamp": timestamp,
                # 嵌套的 coordinates 字典
                "coordinates": {
                    "lat": current_coords.get("lat"),
                    "lng": current_coords.get("lng"),
                    "source": "panorama_object",
                },
                # 顶层的 lat/lng
                "lat": current_coords.get("lat"),
                "lng": current_coords.get("lng"),
                "address": address or "Unknown",
                "source": "panorama_object",
                # 嵌套的 identifiers 字典 (现在填充的是实时数据)
                "identifiers": {
                    "initPanoId": live_identifiers.get("panoId"),  # 实时PanoID
                    "permLink": live_identifiers.get("permLink"),  # 实时链接
                    # 保留旧字段，但填充新数据或留空
                    "initString": live_identifiers.get("urlString"),
                    "locationString": address,
                    "url": live_identifiers.get("permLink"),
                },
                # 顶层的链接字段
                "url": live_identifiers.get("permLink"),
                "init_string": live_identifiers.get("urlString"),
                "pano_id": live_identifiers.get("panoId"),
                "perm_link": live_identifiers.get("permLink"),
                "collection_options": self.options.copy(),
            }

            # 保存缩略图
            if DATA_COLLECTION_CONFIG.get("save_thumbnails", True):
                thumbnail_path = self.save_thumbnail(sample_id)
                if thumbnail_path:
                    location_data["thumbnail_path"] = thumbnail_path
                    location_data["has_thumbnail"] = True
                else:
                    location_data["has_thumbnail"] = False

            return location_data

        except Exception as e:
            print(f"❌ Error in collect_single_location: {e}")
            return None

    # ... (save_thumbnail, save_data 等其他函数保持不变) ...
    def save_thumbnail(self, sample_id: str) -> Optional[str]:
        try:
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if not screenshot_bytes:
                return None
            image = Image.open(BytesIO(screenshot_bytes))
            thumbnail_size = DATA_COLLECTION_CONFIG.get("thumbnail_size", (320, 240))
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            thumbnail_filename = f"{sample_id}.jpg"
            thumbnail_path = os.path.join(DATA_PATHS["thumbnails"], thumbnail_filename)
            if image.mode in ("RGBA", "LA"):
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                image = rgb_image
            image.save(thumbnail_path, "JPEG", quality=85)
            return thumbnail_filename
        except Exception:
            return None

    def save_data(self):
        try:
            output_data = {
                "metadata": {"collection_date": datetime.now().isoformat()},
                "samples": self.data,
            }
            with open(DATA_PATHS["golden_labels"], "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Location data saved to {DATA_PATHS['golden_labels']}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def close(self):
        self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
