# data_collector.py (Final Version for High-Quality Data)

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
        num_samples = num_samples or BENCHMARK_CONFIG["data_collection_samples"]
        print(f"🚀 Starting high-quality data collection for {num_samples} samples...")

        # NOTE: setup_collection_options is not implemented in the provided controller, assuming it's handled manually or not needed.

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

    # 在 data_collector.py 中替换此函数

    def collect_single_location(self) -> Optional[Dict]:
        """Collects a single location and manually constructs the url_slug."""
        try:
            # 1. 获取坐标和标识符
            coords = self.controller.driver.execute_script(
                "return { lat: window.panorama.getPosition().lat(), lng: window.panorama.getPosition().lng() };"
            )
            if not coords:
                raise ValueError("Could not get coordinates.")

            identifiers = self.controller.get_live_location_identifiers()
            if not identifiers or "pov" not in identifiers:
                raise ValueError("Could not get POV.")

            address = self.controller.get_current_address()

            # **2. 核心修复：在Python中手动构建url_slug**
            lat = coords.get("lat")
            lng = coords.get("lng")
            pov = identifiers.get("pov")
            # MapCrunch的URL slug中，zoom是0-based，而Google POV是1-based
            zoom_for_slug = round(pov.get("zoom", 1.0)) - 1

            # 使用 roundNum 函数的逻辑来格式化数字
            def round_num(n, d):
                return f"{n:.{d}f}"

            url_slug = (
                f"{round_num(lat, 6)}_"
                f"{round_num(lng, 6)}_"
                f"{round_num(pov.get('heading', 0), 2)}_"
                f"{round_num(pov.get('pitch', 0) * -1, 2)}_"  # Pitch在slug中是负数
                f"{zoom_for_slug}"
            )

            # 3. 构建数据样本
            sample_id = str(uuid.uuid4())
            location_data = {
                "id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "lat": lat,
                "lng": lng,
                "address": address or "Unknown",
                "pano_id": identifiers.get("panoId"),
                "pov": pov,
                "url_slug": url_slug,  # <-- 现在这里永远有正确的值
            }

            # 4. 保存缩略图
            thumbnail_path = self.save_thumbnail(sample_id)
            if thumbnail_path:
                location_data["thumbnail_path"] = thumbnail_path

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
                "metadata": {
                    "collection_date": datetime.now().isoformat(),
                    "collection_options": self.options,
                },
                "samples": self.data,
            }
            with open(DATA_PATHS["golden_labels"], "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 High-quality data saved to {DATA_PATHS['golden_labels']}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def close(self):
        self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
