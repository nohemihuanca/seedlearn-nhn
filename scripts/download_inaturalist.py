#!/usr/bin/env python3
"""
Robust download script for Yale-STRI AI Seedling Project with better error handling and progress tracking.
"""

import requests
import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict
import sys

class RobustProjectDownloader:
    def __init__(self, project_id=228504, data_dir="/gpfs/gibbs/project/yse/mjh225/repos/seedlearn/data/iNaturalist-082025"):
        self.project_id = project_id
        self.base_url = "https://api.inaturalist.org/v1"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_observations': 0,
            'processed_observations': 0,
            'total_photos': 0,
            'downloaded_photos': 0,
            'skipped_photos': 0,
            'failed_photos': 0,
            'failed_photo_ids': []
        }
        
        self.log_file = self.data_dir / f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log(self, message, force_print=False):
        """Log to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        # Always print progress messages
        if force_print or "Progress:" in message or "ERROR" in message or "Complete" in message:
            print(log_message)
            sys.stdout.flush()
            
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
            f.flush()
            
    def download_photo_with_retry(self, photo_data, obs_dir, max_retries=3):
        """Download a photo with retry logic and multiple URL attempts."""
        photo_id = photo_data.get('id')
        
        # Skip if already downloaded
        existing_files = list(obs_dir.glob(f"photo_{photo_id}*.jpg"))
        if existing_files:
            self.stats['skipped_photos'] += 1
            return True
            
        # Try different URL patterns
        url_patterns = [
            f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.jpg",
            f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/large.jpg",
            f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/medium.jpg",
        ]
        
        # Also try the URL from the photo data if available
        if photo_data.get('url'):
            # Get the original size URL
            base_url = photo_data['url'].replace('/square.', '/original.')
            url_patterns.insert(0, base_url)
            
        for url in url_patterns:
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=30, stream=True)
                    if response.status_code == 200:
                        # Save the photo
                        filename = f"photo_{photo_id}.jpg"
                        filepath = obs_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                
                        self.stats['downloaded_photos'] += 1
                        
                        # Save metadata
                        metadata = {
                            'photo_id': photo_id,
                            'url_used': url,
                            'attribution': photo_data.get('attribution', ''),
                            'license_code': photo_data.get('license_code', '')
                        }
                        
                        metadata_file = obs_dir / f"photo_{photo_id}_metadata.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                            
                        return True
                        
                    elif response.status_code == 404:
                        # Try next URL pattern
                        break
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    else:
                        self.log(f"Failed to download photo {photo_id} from {url}: {e}")
                        
        # If all attempts failed
        self.stats['failed_photos'] += 1
        self.stats['failed_photo_ids'].append(photo_id)
        return False
        
    def process_observation(self, observation):
        """Process a single observation."""
        obs_id = observation.get('id')
        username = observation.get('user', {}).get('login', 'unknown')
        photos = observation.get('photos', [])
        
        if not photos:
            return
            
        # Create directory
        obs_dir = self.data_dir / f"project_{self.project_id}" / username / str(obs_id)
        obs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save observation metadata
        obs_metadata = {
            'observation_id': obs_id,
            'username': username,
            'species_guess': observation.get('species_guess'),
            'taxon': observation.get('taxon', {}),
            'observed_on': observation.get('observed_on'),
            'description': observation.get('description', ''),
            'place_guess': observation.get('place_guess'),
            'quality_grade': observation.get('quality_grade'),
            'num_photos': len(photos),
            'project_id': self.project_id
        }
        
        metadata_file = obs_dir / "observation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(obs_metadata, f, indent=2)
            
        # Download photos
        for photo in photos:
            self.stats['total_photos'] += 1
            self.download_photo_with_retry(photo, obs_dir)
            
    def download_project(self):
        """Download all observations from the project."""
        self.log(f"Starting download for project {self.project_id}", force_print=True)
        
        page = 1
        per_page = 200
        
        # First, get total count
        params = {'project_id': self.project_id, 'per_page': 1}
        try:
            response = requests.get(f"{self.base_url}/observations", params=params)
            data = response.json()
            total_observations = data.get('total_results', 0)
            self.stats['total_observations'] = total_observations
            self.log(f"Total observations in project: {total_observations}", force_print=True)
        except Exception as e:
            self.log(f"ERROR getting observation count: {e}", force_print=True)
            return
            
        # Download all pages
        while True:
            params = {
                'project_id': self.project_id,
                'page': page,
                'per_page': per_page,
                'order': 'desc',
                'order_by': 'created_at'
            }
            
            try:
                self.log(f"Fetching page {page}...")
                response = requests.get(f"{self.base_url}/observations", params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                if not results:
                    break
                    
                # Process each observation
                for i, obs in enumerate(results):
                    self.stats['processed_observations'] += 1
                    self.process_observation(obs)
                    
                    # Progress update every 10 observations
                    if self.stats['processed_observations'] % 10 == 0:
                        progress_pct = (self.stats['processed_observations'] / total_observations) * 100
                        self.log(
                            f"Progress: {self.stats['processed_observations']}/{total_observations} observations "
                            f"({progress_pct:.1f}%), "
                            f"{self.stats['downloaded_photos']} photos downloaded, "
                            f"{self.stats['failed_photos']} failed",
                            force_print=True
                        )
                        
                page += 1
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.log(f"ERROR on page {page}: {e}", force_print=True)
                time.sleep(5)  # Wait before retrying
                
        # Save final statistics
        self.save_statistics()
        
    def save_statistics(self):
        """Save download statistics."""
        self.stats['download_timestamp'] = datetime.now().isoformat()
        self.stats['success_rate'] = (
            (self.stats['downloaded_photos'] / self.stats['total_photos'] * 100)
            if self.stats['total_photos'] > 0 else 0
        )
        
        stats_file = self.data_dir / "download_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        # Print summary
        self.log("\n" + "="*60, force_print=True)
        self.log("DOWNLOAD COMPLETE", force_print=True)
        self.log("="*60, force_print=True)
        self.log(f"Observations processed: {self.stats['processed_observations']}/{self.stats['total_observations']}", force_print=True)
        self.log(f"Total photos: {self.stats['total_photos']}", force_print=True)
        self.log(f"Downloaded: {self.stats['downloaded_photos']}", force_print=True)
        self.log(f"Skipped (already existed): {self.stats['skipped_photos']}", force_print=True)
        self.log(f"Failed: {self.stats['failed_photos']}", force_print=True)
        self.log(f"Success rate: {self.stats['success_rate']:.1f}%", force_print=True)
        self.log(f"\nStatistics saved to: {stats_file}", force_print=True)


def main():
    downloader = RobustProjectDownloader()
    downloader.download_project()


if __name__ == "__main__":
    main()