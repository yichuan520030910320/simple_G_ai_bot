#!/usr/bin/env python3
"""
Quick viewer for collected MapCrunch data
"""

import json
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter

def view_data_summary(data_file='data/golden_labels.json'):
    """Display summary of collected data"""
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ No data file found at {data_file}")
        print("💡 Run data collection first: python main.py --mode data --samples 50")
        return
    
    samples = data.get('samples', [])
    metadata = data.get('metadata', {})
    
    print(f"📊 MapCrunch Data Collection Summary")
    print(f"{'='*50}")
    print(f"📅 Collection Date: {metadata.get('collection_date', 'Unknown')}")
    print(f"📍 Total Samples: {len(samples)}")
    print(f"🏙️  Collection Options: {metadata.get('collection_options', {})}")
    
    # Statistics
    stats = metadata.get('statistics', {})
    if stats:
        print(f"\n📈 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Country distribution
    countries = []
    for sample in samples:
        address = sample.get('address', '')
        if address and address != 'Unknown':
            # Extract country (usually last part after comma)
            country = address.split(', ')[-1].strip()
            countries.append(country)
    
    if countries:
        country_counts = Counter(countries)
        print(f"\n🌍 Top Countries:")
        for country, count in country_counts.most_common(10):
            print(f"   {country}: {count} samples")
    
    # Coordinate coverage
    coords_available = sum(1 for s in samples if s.get('lat') is not None)
    print(f"\n📍 Coordinate Coverage: {coords_available}/{len(samples)} ({coords_available/len(samples)*100:.1f}%)")
    
    # Thumbnail coverage
    thumbnails_available = sum(1 for s in samples if s.get('has_thumbnail'))
    print(f"📸 Thumbnail Coverage: {thumbnails_available}/{len(samples)} ({thumbnails_available/len(samples)*100:.1f}%)")
    
    # Sample locations
    print(f"\n📍 Sample Locations:")
    for i, sample in enumerate(samples[:10]):
        address = sample.get('address', 'Unknown')
        lat = sample.get('lat', 'N/A')
        lng = sample.get('lng', 'N/A')
        has_thumb = "📸" if sample.get('has_thumbnail') else "❌"
        print(f"   {i+1}. {has_thumb} {address} ({lat}, {lng})")
    
    if len(samples) > 10:
        print(f"   ... and {len(samples) - 10} more")


def create_thumbnail_gallery(data_file='data/golden_labels.json', output_file='data/gallery.html', max_images=100):
    """Create an HTML gallery of collected thumbnails"""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    
    html = """
    <html>
    <head>
        <title>MapCrunch Collection Gallery</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f0f0; }
            h1 { text-align: center; }
            .gallery { display: flex; flex-wrap: wrap; justify-content: center; }
            .item { 
                margin: 10px; 
                background: white; 
                padding: 10px; 
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .item img { max-width: 320px; border-radius: 4px; }
            .address { font-weight: bold; margin: 5px 0; }
            .coords { font-size: 0.9em; color: #666; }
            .stats { margin: 20px; text-align: center; }
        </style>
    </head>
    <body>
        <h1>MapCrunch Collection Gallery</h1>
    """
    
    # Add statistics
    total = len(samples)
    with_thumb = sum(1 for s in samples if s.get('has_thumbnail'))
    with_coords = sum(1 for s in samples if s.get('lat') is not None)
    
    html += f"""
        <div class="stats">
            <p>Total Samples: {total} | With Thumbnails: {with_thumb} | With Coordinates: {with_coords}</p>
        </div>
        <div class="gallery">
    """
    
    # Add thumbnails
    count = 0
    for sample in samples:
        if count >= max_images:
            break
            
        if sample.get('thumbnail_path'):
            thumb_path = f"thumbnails/{sample['thumbnail_path']}"
            address = sample.get('address', 'Unknown')
            lat = sample.get('lat', 'N/A')
            lng = sample.get('lng', 'N/A')
            
            html += f"""
            <div class="item">
                <img src="{thumb_path}" alt="{address}">
                <div class="address">{address}</div>
                <div class="coords">{lat}, {lng}</div>
            </div>
            """
            count += 1
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"✅ Gallery created: {output_file}")
    print(f"📸 Included {count} images")
    print(f"💡 Open in browser: file://{os.path.abspath(output_file)}")


def plot_thumbnails_grid(data_file='data/golden_labels.json', max_images=20):
    """Display a grid of thumbnails using matplotlib"""
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    samples = [s for s in data['samples'] if s.get('thumbnail_path')][:max_images]
    
    if not samples:
        print("❌ No samples with thumbnails found")
        return
    
    # Create grid
    cols = 5
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        row = i // cols
        col = i % cols
        
        thumb_path = f"data/thumbnails/{sample['thumbnail_path']}"
        if os.path.exists(thumb_path):
            img = mpimg.imread(thumb_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(sample.get('address', 'Unknown')[:30] + '...', fontsize=8)
        
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(samples), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'MapCrunch Collection Sample ({len(samples)} locations)', y=1.02)
    plt.show()


def export_coordinates_csv(data_file='data/golden_labels.json', output_file='data/coordinates.csv'):
    """Export coordinates to CSV for mapping"""
    
    import csv
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'address', 'latitude', 'longitude', 'has_thumbnail'])
        
        count = 0
        for sample in samples:
            if sample.get('lat') is not None and sample.get('lng') is not None:
                writer.writerow([
                    sample['id'][:8],
                    sample.get('address', 'Unknown'),
                    sample['lat'],
                    sample['lng'],
                    'Yes' if sample.get('has_thumbnail') else 'No'
                ])
                count += 1
    
    print(f"✅ Exported {count} coordinates to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='View collected MapCrunch data')
    parser.add_argument('--gallery', action='store_true', help='Create HTML gallery')
    parser.add_argument('--grid', action='store_true', help='Show thumbnail grid')
    parser.add_argument('--csv', action='store_true', help='Export coordinates to CSV')
    parser.add_argument('--data', default='data/golden_labels.json', help='Data file path')
    parser.add_argument('--max-images', type=int, default=50, help='Max images for gallery/grid')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("💡 Run data collection first: python main.py --mode data --samples 50")
        return
    
    # Always show summary
    view_data_summary(args.data)
    
    # Additional actions
    if args.gallery:
        print()
        create_thumbnail_gallery(args.data, max_images=args.max_images)
    
    if args.grid:
        print()
        plot_thumbnails_grid(args.data, max_images=args.max_images)
    
    if args.csv:
        print()
        export_coordinates_csv(args.data)


if __name__ == "__main__":
    main()