#!/usr/bin/env python3
"""
ML-optimized sorting script for iNaturalist seedling images.
Implements hierarchical directory structure for flexible deep learning training.
Separates verification images and provides comprehensive taxonomy normalization.
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Optional

class MLProjectSorter:
    def __init__(self,
                 source_dir="/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/inaturalist/project_228504",
                 base_output_dir="/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/raw",
                 species_csv="YPS_seedling_spp_list_code_10_22_25.csv"):
        
        self.source_dir = Path(source_dir)
        self.species_csv = Path(species_csv)
        
        # Create date-based output directory structure
        today = datetime.now().strftime("%Y-%m-%d")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version = f"v{today}_12K"
        self.dest_base = Path(base_output_dir) / today / "sorted_12K"
        self.training_dir = self.dest_base / "training"
        self.verification_dir = self.dest_base / "verification"
        self.metadata_dir = self.dest_base / "metadata"
        
        # Create all directories
        for directory in [self.training_dir, self.verification_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Define metadata files (removed redundant taxonomy_mapping and verification_index)
        self.log_files = {
            'sorting': self.metadata_dir / "sorting_log.json",
            'normalization': self.metadata_dir / "normalization_log.csv",
            'summary': self.metadata_dir / "processing_summary.json",
            'enhanced_species': self.metadata_dir / f"species_catalog_{self.version}_{self.timestamp}.csv",
            'all_issues': self.metadata_dir / f"all_issues_{self.version}_{self.timestamp}.csv"
        }
        
        # Initialize logging structures
        self.sorting_log = []
        self.normalization_log = []
        self.all_issues = []  # Unified issue tracking
        
        # Initialize CSV headers
        self._init_csv_headers()
        
        # User-specific tag positions
        self.user_tag_positions = {
            'bianco': 'last',
            'biancolini23': 'last',
            'crono_secuencia5': 'first',
            'crono_secuencia_2': 'last',
            'crono_secuencia_3': 'last',
            'crono_secuencia_4': 'first',
            'maria': 'last'
        }
        
        # Initialize empty corrections dictionary (can be used for future corrections if needed)
        self.taxonomy_corrections = {
            'genus': {},
            'family': {}
        }
        
        # Initialize statistics
        self.stats = {
            'total_observations': 0,
            'processed_observations': 0,
            'training_images': 0,
            'verification_images': 0,
            'individuals_skipped_min_images': 0,
            'taxonomy_normalizations': 0,
            'duplicate_ids_found': 0,
            'corrupted_images': 0,
            'missing_taxonomy': 0,
            'users_processed': set(),
            'unique_families': set(),
            'unique_genera': set(),
            'unique_species': set(),
            'unique_individuals': set()
        }
        
        # Track image counts per individual
        self.individual_img_counts = defaultdict(int)
        
        # Track enhanced species information
        self.enhanced_species_data = {}
        
        # Load species taxonomy
        self.id_to_taxonomy = {}
        self._load_species_data()
        
        self._log_info(f"ML Project Sorter initialized")
        self._log_info(f"Output directory: {self.dest_base}")
        
    def _init_csv_headers(self):
        """Initialize CSV files with headers."""
        # Normalization log
        with open(self.log_files['normalization'], 'w') as f:
            f.write("timestamp,field,original,normalized,reason,confidence\n")
            
        # Unified issues log
        with open(self.log_files['all_issues'], 'w') as f:
            f.write("timestamp,severity,issue_type,user,observation_id,individual_id,description,action_needed,file_count,expected_count,can_process\n")
            
    def _log_info(self, message: str):
        """Log informational messages."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def _log_issue(self, severity: str, issue_type: str, user: str, observation_id: str, 
                   individual_id: str, description: str, action_needed: str, 
                   file_count: int = None, expected_count: int = None, can_process: bool = False):
        """Log issues to unified tracking system."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to in-memory list
        issue = {
            'timestamp': timestamp,
            'severity': severity,  # ERROR, WARNING, INFO
            'issue_type': issue_type,
            'user': user,
            'observation_id': observation_id,
            'individual_id': individual_id or 'N/A',
            'description': description,
            'action_needed': action_needed,
            'file_count': file_count or 'N/A',
            'expected_count': expected_count or 'N/A',
            'can_process': can_process
        }
        self.all_issues.append(issue)
        
        # Write immediately to CSV
        with open(self.log_files['all_issues'], 'a') as f:
            f.write(f"{timestamp},{severity},{issue_type},{user},{observation_id},"
                   f"{individual_id or 'N/A'},{description},{action_needed},"
                   f"{file_count or 'N/A'},{expected_count or 'N/A'},{can_process}\n")
        
        # Also log to console
        self._log_info(f"[{severity}] {issue_type}: {user}/{observation_id} - {description}")
        
    def _normalize_taxonomy_name(self, name: str, level: str) -> Tuple[str, List[Dict]]:
        """
        Normalize taxonomy names according to rules.
        Returns normalized name and list of normalization actions.
        """
        if not name or name.strip() == '':
            normalized = f"Unknown_{level.title()}" if level != 'species' else "unknown_species"
            return normalized, [{'reason': 'missing_value', 'confidence': 1.0}]
            
        original = name.strip()
        normalized = original
        actions = []
        
        # Skip corrections - now empty
        # if level in self.taxonomy_corrections and normalized.lower() in self.taxonomy_corrections[level]:
        #     normalized = self.taxonomy_corrections[level][normalized.lower()]
        #     actions.append({'reason': 'known_variation', 'confidence': 1.0})
            
        # Remove special characters
        if re.search(r'[^a-zA-Z0-9\s_-]', normalized):
            normalized = re.sub(r'[^a-zA-Z0-9\s_-]', '_', normalized)
            actions.append({'reason': 'special_characters', 'confidence': 0.9})
            
        # Handle spaces
        if ' ' in normalized:
            normalized = normalized.replace(' ', '_')
            actions.append({'reason': 'spaces_to_underscores', 'confidence': 1.0})
            
        # Multiple underscores to single
        if '__' in normalized:
            normalized = re.sub(r'_+', '_', normalized)
            actions.append({'reason': 'multiple_underscores', 'confidence': 1.0})
            
        # Apply capitalization rules
        if level == 'family' or level == 'genus':
            # Title case for family and genus
            if not normalized[0].isupper() or any(c.isupper() for c in normalized[1:]):
                parts = normalized.split('_')
                normalized = '_'.join(word.capitalize() for word in parts)
                actions.append({'reason': 'capitalization', 'confidence': 1.0})
        elif level == 'species':
            # Lowercase for species
            if any(c.isupper() for c in normalized):
                normalized = normalized.lower()
                actions.append({'reason': 'lowercase', 'confidence': 1.0})
                
        # Log normalization if changed
        if normalized != original:
            self.stats['taxonomy_normalizations'] += 1
            for action in actions:
                self._log_normalization(level, original, normalized, action['reason'], action['confidence'])
                
        return normalized, actions
        
    def _log_normalization(self, field: str, original: str, normalized: str, reason: str, confidence: float):
        """Log taxonomy normalization to CSV."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_files['normalization'], 'a') as f:
            f.write(f"{timestamp},{field},{original},{normalized},{reason},{confidence}\n")
            
        self.normalization_log.append({
            'timestamp': timestamp,
            'field': field,
            'original': original,
            'normalized': normalized,
            'reason': reason,
            'confidence': confidence
        })
        
    def _load_species_data(self):
        """Load species taxonomy from CSV."""
        if not self.species_csv.exists():
            self._log_info(f"ERROR: Species CSV not found at {self.species_csv}")
            raise FileNotFoundError(f"Species CSV required at {self.species_csv}")
            
        self._log_info(f"Loading species data from: {self.species_csv}")
        
        try:
            species_df = pd.read_csv(self.species_csv)
            
            # Check required columns
            required_cols = ['ID_YPS', 'SPP', 'GENUS', 'SPECIES', 'FAMILY']
            missing_cols = [col for col in required_cols if col not in species_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Process each row
            for _, row in species_df.iterrows():
                # Normalize taxonomy names
                family, _ = self._normalize_taxonomy_name(str(row['FAMILY']), 'family')
                genus, _ = self._normalize_taxonomy_name(str(row['GENUS']), 'genus')
                species, _ = self._normalize_taxonomy_name(str(row['SPECIES']), 'species')
                
                mapping_data = {
                    'family': family,
                    'genus': genus,
                    'species': species,
                    'spp': row['SPP'],
                    'id_yps': row['ID_YPS'],
                    'liana': row.get('LIANA', 0),
                    'forest': row.get('FOREST', 'Unknown')
                }
                
                # Map using ID_YPS
                id_yps_key = str(row['ID_YPS']).lower().strip()
                self.id_to_taxonomy[id_yps_key] = mapping_data
                
                # Also map variations
                if id_yps_key.startswith(('pp', 'bar', 'pea')):
                    if len(id_yps_key) > 3:
                        suffix = id_yps_key[3:] if id_yps_key.startswith('bar') or id_yps_key.startswith('pea') else id_yps_key[2:]
                        self.id_to_taxonomy[suffix] = mapping_data
                        
                # Map SPP code
                if pd.notna(row['SPP']):
                    spp_key = str(row['SPP']).lower().strip()
                    if spp_key not in self.id_to_taxonomy:
                        self.id_to_taxonomy[spp_key] = mapping_data
                        
            self._log_info(f"Loaded taxonomy for {len(species_df)} individuals")
            self._log_info(f"Created {len(self.id_to_taxonomy)} ID mappings")
            
            # Save a copy of the species CSV to metadata
            species_copy = self.metadata_dir / f"species_csv_snapshot.csv"
            species_df.to_csv(species_copy, index=False)
                    
        except Exception as e:
            self._log_info(f"ERROR loading species data: {str(e)}")
            raise
            
    def _find_taxonomy_info(self, individual_id: str) -> Optional[Dict]:
        """Find taxonomy info for individual - STRICT MATCHING ONLY.

        Only allows:
        1. Exact match (case-insensitive)
        2. Punctuation normalization (., -, _ removed)

        NO fuzzy/substring matching to prevent data contamination.
        """
        id_lower = individual_id.lower().strip()

        # 1. Direct exact match (case-insensitive)
        if id_lower in self.id_to_taxonomy:
            return self.id_to_taxonomy[id_lower]

        # 2. Try with punctuation normalization
        id_clean = id_lower.replace('.', '').replace('-', '').replace('_', '')
        if id_clean in self.id_to_taxonomy:
            self._log_info(f"Matched {individual_id} via punctuation normalization -> {id_clean}")
            return self.id_to_taxonomy[id_clean]

        # 3. NO FUZZY MATCHING - Return None and let caller log the issue
        return None
        
    def _create_training_path(self, taxonomy_info: Dict) -> Path:
        """Create the training directory path for an individual."""
        family = taxonomy_info['family']
        genus = taxonomy_info['genus']
        species = taxonomy_info['species']
        individual_id = taxonomy_info['id_yps']
        
        path = (self.training_dir / "by_family" / family / 
                "by_genus" / genus / 
                "by_species" / species / individual_id)
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    def _create_verification_path(self, taxonomy_info: Dict) -> Path:
        """Create the verification directory path for an individual."""
        family = taxonomy_info['family']
        genus = taxonomy_info['genus']
        species = taxonomy_info['species']
        individual_id = taxonomy_info['id_yps']
        
        path = self.verification_dir / family / genus / species / individual_id
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    def _validate_image(self, image_path: Path) -> bool:
        """Validate that an image file is not corrupted."""
        try:
            # Check file size (100KB to 10MB)
            size = image_path.stat().st_size
            if size < 100 * 1024 or size > 10 * 1024 * 1024:
                return False
                
            # Could add more validation here (e.g., try to open with PIL)
            return True
            
        except Exception:
            return False
            
    def _check_observation_files(self, obs_dir: Path, username: str, observation_id: str):
        """Check observation directory for file count issues."""
        files = list(obs_dir.iterdir())
        file_count = len(files)
        
        jpg_files = [f for f in files if f.suffix == '.jpg']
        json_files = [f for f in files if f.suffix == '.json']
        
        # Expected: 13 files (6 JPG + 7 JSON) or 11 files (5 JPG + 6 JSON)
        if file_count == 13 and len(jpg_files) == 6 and len(json_files) == 7:
            # Normal, complete observation
            pass
        elif file_count == 11 and len(jpg_files) == 5 and len(json_files) == 6:
            # Known issue: missing one image pair
            self._log_issue("INFO", "INCOMPLETE_SET", username, observation_id, None,
                          "Missing 1 image and metadata pair (11 files instead of 13)",
                          "Expected variation - can still process", 
                          file_count=11, expected_count=13, can_process=True)
        else:
            # Unexpected file count
            self._log_issue("WARNING", "UNEXPECTED_FILE_COUNT", username, observation_id, None,
                          f"Unexpected file structure: {file_count} files ({len(jpg_files)} JPG, {len(json_files)} JSON)",
                          "Verify download completed successfully",
                          file_count=file_count, expected_count=13, can_process=True)
    
    def process_observations(self):
        """Process all observations with ML-optimized sorting."""
        self._log_info("Starting ML-optimized observation processing...")
        self._log_info(f"Training images will be stored in: {self.training_dir}")
        self._log_info(f"Verification images will be stored in: {self.verification_dir}")
        
        # Process each user
        for username in sorted(os.listdir(self.source_dir)):
            user_dir = self.source_dir / username
            if not user_dir.is_dir():
                continue
                
            self._log_info(f"\nProcessing user: {username}")
            self.stats['users_processed'].add(username)
            
            tag_position = self.user_tag_positions.get(username, 'last')
            self._log_info(f"  Tag position for {username}: {tag_position}")
            
            user_obs_count = 0
            user_processed = 0
            
            for observation_id in os.listdir(user_dir):
                obs_dir = user_dir / observation_id
                if not obs_dir.is_dir():
                    continue
                    
                user_obs_count += 1
                self.stats['total_observations'] += 1
                
                # Check file count before processing
                self._check_observation_files(obs_dir, username, observation_id)
                
                if self._process_observation(obs_dir, username, observation_id, tag_position):
                    user_processed += 1
                    
            self._log_info(f"  User {username}: {user_processed}/{user_obs_count} observations processed")
            
        # Save final results
        self._save_final_results()
        
    def _process_observation(self, obs_dir: Path, username: str, observation_id: str, tag_position: str) -> bool:
        """Process single observation with ML sorting."""
        try:
            # Load metadata
            obs_metadata_path = obs_dir / 'observation_metadata.json'
            if not obs_metadata_path.exists():
                self._log_issue("ERROR", "NO_METADATA", username, observation_id, None,
                              "Missing observation_metadata.json file",
                              "Check if file was downloaded correctly", can_process=False)
                return False
                
            with open(obs_metadata_path, 'r') as f:
                obs_metadata = json.load(f)
                
            # Get ID from description field
            individual_id = obs_metadata.get('description', '').strip()
            if not individual_id:
                self._log_issue("ERROR", "NO_ID", username, observation_id, None,
                              "No individual ID found in description field",
                              "Add ID to observation description on iNaturalist", can_process=False)
                return False
                
            # Find taxonomy
            taxonomy_info = self._find_taxonomy_info(individual_id)
            
            if not taxonomy_info:
                self._log_issue("ERROR", "MISSING_TAXONOMY", username, observation_id, individual_id,
                              f"No taxonomy mapping found for ID: {individual_id}",
                              "Add this ID to the species CSV file", can_process=False)
                self.stats['missing_taxonomy'] += 1
                return False
                
            # Get photos
            photos = self._get_sorted_photos(obs_dir)
            
            if len(photos) == 0:
                self._log_issue("ERROR", "NO_IMAGES", username, observation_id, individual_id,
                              "No JPG images found in observation",
                              "Check if images were downloaded correctly", can_process=False)
                return False
                
            # Check minimum image threshold
            training_image_count = len(photos) - 1  # Exclude verification image
            if training_image_count < 3:
                self._log_issue("WARNING", "INSUFFICIENT_IMAGES", username, observation_id, individual_id,
                              f"Only {training_image_count} training images (minimum 3 required)",
                              "Capture more images for this individual", 
                              file_count=len(photos), expected_count=4, can_process=False)
                self.stats['individuals_skipped_min_images'] += 1
                return False
                
            # Process photos with ML structure
            training_path, verification_info = self._sort_photos_ml(photos, obs_dir, obs_metadata, 
                               taxonomy_info, username, observation_id, tag_position)
            
            # Track enhanced species data
            self._update_enhanced_species_data(taxonomy_info, training_path, training_image_count, 
                                             verification_info, username)
            
            self.stats['processed_observations'] += 1
            return True
            
        except Exception as e:
            self._log_issue("ERROR", "PROCESSING_ERROR", username, observation_id, 
                          individual_id if 'individual_id' in locals() else None,
                          f"Unexpected error: {str(e)}",
                          "Review error details and contact support if needed", can_process=False)
            return False
            
    def _get_sorted_photos(self, obs_dir: Path) -> List[Tuple[int, str]]:
        """Get photos sorted by photo ID."""
        photos = []
        for file_name in os.listdir(obs_dir):
            if file_name.endswith('.jpg'):
                try:
                    photo_id = int(file_name.split('_')[1].split('.')[0])
                    photos.append((photo_id, file_name))
                except:
                    pass  # Ignore files that don't match expected pattern
        return sorted(photos, key=lambda x: x[0])
        
    def _sort_photos_ml(self, photos: List[Tuple[int, str]], obs_dir: Path, obs_metadata: Dict,
                        taxonomy_info: Dict, username: str, observation_id: str, tag_position: str) -> Tuple[Path, Dict]:
        """Sort photos with ML-optimized structure. Returns training path and verification info."""
        # Create paths
        training_path = self._create_training_path(taxonomy_info)
        verification_path = self._create_verification_path(taxonomy_info)
        
        # Track unique individuals
        individual_key = f"{taxonomy_info['family']}/{taxonomy_info['genus']}/{taxonomy_info['species']}/{taxonomy_info['id_yps']}"
        self.stats['unique_individuals'].add(individual_key)
        self.stats['unique_families'].add(taxonomy_info['family'])
        self.stats['unique_genera'].add(f"{taxonomy_info['family']}/{taxonomy_info['genus']}")
        self.stats['unique_species'].add(f"{taxonomy_info['family']}/{taxonomy_info['genus']}/{taxonomy_info['species']}")
        
        # Initialize counter for this individual if needed
        if individual_key not in self.individual_img_counts:
            self.individual_img_counts[individual_key] = 0
            
        # Save individual metadata once
        individual_metadata_path = training_path / 'individual_metadata.json'
        if not individual_metadata_path.exists():
            metadata = {
                **obs_metadata,
                'taxonomy': taxonomy_info,
                'normalized_taxonomy': {
                    'family': taxonomy_info['family'],
                    'genus': taxonomy_info['genus'],
                    'species': taxonomy_info['species']
                },
                'attributes': {
                    'liana': taxonomy_info.get('liana', 0),
                    'forest': taxonomy_info.get('forest', 'Unknown')
                }
            }
            with open(individual_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # Determine verification image index
        verification_idx = 0 if tag_position == 'first' else len(photos) - 1
        verification_info = {'path': None, 'position': tag_position}
        
        # Process photos
        training_count = 0
        for idx, (photo_id, filename) in enumerate(photos):
            src_path = obs_dir / filename
            
            # Validate image
            if not self._validate_image(src_path):
                self._log_issue("WARNING", "CORRUPTED_IMAGE", username, observation_id, 
                              taxonomy_info['id_yps'], f"Invalid/corrupted image: {filename}",
                              "Re-download or recapture this image", can_process=True)
                self.stats['corrupted_images'] += 1
                continue
                
            if idx == verification_idx:
                # Process as verification image
                verification_file = self._process_verification_image(
                    src_path, verification_path, taxonomy_info, 
                    username, tag_position, training_count
                )
                verification_info['path'] = str(verification_file)
            else:
                # Process as training image
                self.individual_img_counts[individual_key] += 1
                img_num = self.individual_img_counts[individual_key]
                
                self._process_training_image(
                    src_path, training_path, taxonomy_info, 
                    img_num, observation_id, photo_id
                )
                training_count += 1
                
        return training_path, verification_info
                
    def _process_training_image(self, src_path: Path, dest_dir: Path, 
                                taxonomy_info: Dict, img_num: int, 
                                observation_id: str, photo_id: int):
        """Process a training image."""
        # Create filename with full taxonomy
        family = taxonomy_info['family']
        genus = taxonomy_info['genus']
        species = taxonomy_info['species']
        individual_id = taxonomy_info['id_yps']
        
        new_filename = f"{family}_{genus}_{species}_{individual_id}_{img_num:03d}.jpg"
        dest_path = dest_dir / new_filename
        
        # Copy image
        shutil.copy2(src_path, dest_path)
        self.stats['training_images'] += 1
        
        # Log sorting decision
        self._log_sorting_decision(
            src_path, dest_path, taxonomy_info,
            'training', observation_id, photo_id
        )
        
    def _process_verification_image(self, src_path: Path, dest_dir: Path,
                                   taxonomy_info: Dict, username: str,
                                   position: str, linked_training_count: int) -> Path:
        """Process a verification image. Returns destination path."""
        # Create filename with verification tag
        family = taxonomy_info['family']
        genus = taxonomy_info['genus']
        species = taxonomy_info['species']
        individual_id = taxonomy_info['id_yps']
        
        new_filename = f"{family}_{genus}_{species}_{individual_id}_verification_{username}_{position}.jpg"
        dest_path = dest_dir / new_filename
        
        # Copy image
        shutil.copy2(src_path, dest_path)
        self.stats['verification_images'] += 1
        
        return dest_path
        
    def _log_sorting_decision(self, src_path: Path, dest_path: Path,
                             taxonomy_info: Dict, image_type: str,
                             observation_id: str, photo_id: int):
        """Log each sorting decision."""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'source_path': str(src_path),
            'dest_path': str(dest_path),
            'image_type': image_type,
            'observation_id': observation_id,
            'photo_id': photo_id,
            'taxonomy': taxonomy_info,
            'normalizations': []  # Would be populated if any were applied
        }
        self.sorting_log.append(decision)
        
    def _update_enhanced_species_data(self, taxonomy_info: Dict, training_path: Path, 
                                     image_count: int, verification_info: Dict, username: str):
        """Update enhanced species catalog with training-relevant information."""
        id_yps = taxonomy_info['id_yps']
        
        # Calculate relative paths from sorted_12K directory
        relative_training_path = training_path.relative_to(self.dest_base)
        relative_verification_path = Path(verification_info['path']).relative_to(self.dest_base) if verification_info['path'] else None
        
        self.enhanced_species_data[id_yps] = {
            # Version info
            'data_version': self.version,
            'sorting_timestamp': self.timestamp,
            'source_project': 'project_228504',
            
            # Original taxonomy
            'ID_YPS': id_yps,
            'SPP': taxonomy_info.get('spp', ''),
            'FAMILY': taxonomy_info['family'],
            'GENUS': taxonomy_info['genus'],
            'SPECIES': taxonomy_info['species'],
            
            # Attributes for ML
            'LIANA': taxonomy_info.get('liana', 0),
            'FOREST': taxonomy_info.get('forest', 'Unknown'),
            
            # Training paths
            'training_path': str(relative_training_path),
            'training_absolute_path': str(training_path),
            
            # Verification info
            'verification_path': str(relative_verification_path) if relative_verification_path else '',
            'verification_absolute_path': verification_info['path'] or '',
            'verification_user': username,
            'verification_position': verification_info['position'],
            
            # Image counts
            'training_image_count': image_count,
            'verification_image_count': 1,
            'total_image_count': image_count + 1,
            
            # Quality flags
            'has_minimum_images': image_count >= 3,
            'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def _save_final_results(self):
        """Save all metadata and generate final report."""
        self._log_info("\nSaving results...")
        
        # Save sorting log
        with open(self.log_files['sorting'], 'w') as f:
            json.dump(self.sorting_log, f, indent=2)
            
        # Save enhanced species catalog
        if self.enhanced_species_data:
            enhanced_df = pd.DataFrame(list(self.enhanced_species_data.values()))
            # Sort by family, genus, species, ID for easy reference
            enhanced_df = enhanced_df.sort_values(['FAMILY', 'GENUS', 'SPECIES', 'ID_YPS'])
            enhanced_df.to_csv(self.log_files['enhanced_species'], index=False)
            self._log_info(f"Saved enhanced species catalog with {len(enhanced_df)} individuals")
        
        # Generate issue summary
        issue_summary = {
            'total_issues': len(self.all_issues),
            'by_severity': {},
            'by_type': {},
            'actionable_items': []
        }
        
        for issue in self.all_issues:
            # Count by severity
            severity = issue['severity']
            issue_summary['by_severity'][severity] = issue_summary['by_severity'].get(severity, 0) + 1
            
            # Count by type
            issue_type = issue['issue_type']
            issue_summary['by_type'][issue_type] = issue_summary['by_type'].get(issue_type, 0) + 1
            
            # Collect unique action items
            if issue['action_needed'] not in issue_summary['actionable_items']:
                issue_summary['actionable_items'].append(issue['action_needed'])
        
        # Generate summary
        summary = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'data_version': self.version,
                'source_directory': str(self.source_dir),
                'output_directory': str(self.dest_base),
                'species_csv_used': str(self.species_csv)
            },
            'statistics': {
                'total_observations': self.stats['total_observations'],
                'processed_observations': self.stats['processed_observations'],
                'training_images': self.stats['training_images'],
                'verification_images': self.stats['verification_images'],
                'users_processed': len(self.stats['users_processed']),
                'unique_families': len(self.stats['unique_families']),
                'unique_genera': len(self.stats['unique_genera']),
                'unique_species': len(self.stats['unique_species']),
                'unique_individuals': len(self.stats['unique_individuals'])
            },
            'quality_control': {
                'individuals_skipped_min_images': self.stats['individuals_skipped_min_images'],
                'taxonomy_normalizations': self.stats['taxonomy_normalizations'],
                'missing_taxonomy': self.stats['missing_taxonomy'],
                'corrupted_images': self.stats['corrupted_images']
            },
            'issues_summary': issue_summary,
            'output_files': {
                'training_directory': str(self.training_dir),
                'verification_directory': str(self.verification_dir),
                'species_catalog': str(self.log_files['enhanced_species']),
                'all_issues_log': str(self.log_files['all_issues']),
                'normalization_log': str(self.log_files['normalization'])
            }
        }
        
        with open(self.log_files['summary'], 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print final summary
        self._log_info("\n" + "=" * 80)
        self._log_info("ML SORTING COMPLETE")
        self._log_info("=" * 80)
        self._log_info(f"Total observations processed: {self.stats['processed_observations']}/{self.stats['total_observations']}")
        self._log_info(f"Training images: {self.stats['training_images']}")
        self._log_info(f"Verification images: {self.stats['verification_images']}")
        self._log_info(f"Unique individuals: {len(self.stats['unique_individuals'])}")
        self._log_info(f"Unique species: {len(self.stats['unique_species'])}")
        
        # Issue summary
        if self.all_issues:
            self._log_info(f"\nISSUES FOUND: {len(self.all_issues)} total")
            if issue_summary['by_severity']:
                for severity, count in sorted(issue_summary['by_severity'].items()):
                    self._log_info(f"  {severity}: {count}")
            self._log_info(f"\nReview all issues in: {self.log_files['all_issues'].name}")
        else:
            self._log_info("\nNo issues encountered!")
            
        self._log_info(f"\nAll metadata saved to: {self.metadata_dir}")
        self._log_info("=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML-optimized sorting script for iNaturalist seedling images"
    )
    parser.add_argument('--source', 
                       default="/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/inaturalist/project_228504",
                       help="Source directory")
    parser.add_argument('--output-base',
                       default="/nfs/roberts/project/pi_lsc4/shared/seedlearn/data/raw",
                       help="Base output directory")
    parser.add_argument('--species-csv',
                       default="YPS_seedling_spp_list_code_10_22_25.csv",
                       help="Species CSV file for taxonomy mapping")
    
    args = parser.parse_args()
    
    sorter = MLProjectSorter(
        source_dir=args.source,
        base_output_dir=args.output_base,
        species_csv=args.species_csv
    )
    
    sorter.process_observations()


if __name__ == "__main__":
    main()