"""Tests for differences between BIDS schema versions."""

import pytest
import tempfile
import json
import os
from pathlib import Path

from bids import BIDSLayout
from bids.layout.models import Config


class TestSchemaVersionDifferences:
    """Test that different schema versions handle different features correctly."""
    
    def test_entity_parsing_version_differences(self):
        """Test that tracksys entity (added in v1.9.0) works in newer schemas but fails in v1.8.0."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "test_dataset"
            dataset_dir.mkdir()
            
            # Create minimal dataset with motion file using tracksys entity
            with open(dataset_dir / "dataset_description.json", "w") as f:
                json.dump({"Name": "Test", "BIDSVersion": "1.10.1", "Authors": ["Test"]}, f)
            
            motion_dir = dataset_dir / "sub-01" / "motion"
            motion_dir.mkdir(parents=True)
            (motion_dir / "sub-01_tracksys-imu_motion.tsv").touch()
            
            # Test tracksys query across schema versions
            def test_tracksys_query(config):
                layout = BIDSLayout(dataset_dir, config=[config])
                try:
                    return len(layout.get(tracksys='imu', return_type='filename')) > 0
                except Exception:
                    return False
            
            # tracksys entity was added in v1.9.0 for motion datatype
            assert test_tracksys_query('bids-schema'), "Current schema should support tracksys"
            assert test_tracksys_query({'schema_version': '1.9.0'}), "v1.9.0 should support tracksys (motion added)"
            assert not test_tracksys_query({'schema_version': '1.8.0'}), "v1.8.0 should NOT support tracksys"
    

    def test_motion_datatype_evolution(self):
        """Test that motion datatype (BEP029) support was added in v1.9.0."""
        
        # Motion datatype was added in v1.9.0 according to changelog:
        # v1.9.0: [ENH] Extend BIDS for Motion data (BEP029) #981
        
        config_v190 = Config.load({'schema_version': '1.9.0'})
        config_v180 = Config.load({'schema_version': '1.8.0'})
        
        entities_v190 = {e.name for e in config_v190.entities.values()}
        entities_v180 = {e.name for e in config_v180.entities.values()}
        
        # Motion-specific entity 'tracksys' should be in v1.9.0+ but not v1.8.0
        assert 'tracksys' in entities_v190, "tracksys entity should exist in v1.9.0 (motion datatype was added)"
        assert 'tracksys' not in entities_v180, "tracksys entity should NOT exist in v1.8.0 (before motion datatype)"
        
        print(f"✓ Motion datatype evolution verified:")
        print(f"  v1.8.0 (before motion): tracksys = {'tracksys' in entities_v180}")
        print(f"  v1.9.0 (motion added): tracksys = {'tracksys' in entities_v190}")
        
        # Test specific motion-related entities that were added
        motion_entities = {'tracksys'}  # Could expand this list as more motion entities are identified
        
        for entity in motion_entities:
            assert entity in entities_v190, f"Motion entity '{entity}' should exist in v1.9.0+"
            assert entity not in entities_v180, f"Motion entity '{entity}' should NOT exist in v1.8.0"
    
    def test_schema_version_metadata_differences(self):
        """Test that schema versions have different BIDS version numbers."""
        
        # Load different schema versions  
        config_current = Config.load('bids-schema')
        config_v190 = Config.load({'schema_version': '1.9.0'})
        config_v180 = Config.load({'schema_version': '1.8.0'})
        
        # Check that config names reflect the versions
        assert 'bids-schema-' in config_current.name
        assert '1.9.0' in config_v190.name
        assert '1.8.0' in config_v180.name
        
        print(f"Current config: {config_current.name}")
        print(f"v1.9.0 config: {config_v190.name}")
        print(f"v1.8.0 config: {config_v180.name}")
        
        # Verify they're different configurations
        assert config_current.name != config_v190.name
        assert config_v190.name != config_v180.name


if __name__ == '__main__':
    # Allow running as script for quick testing
    test_instance = TestSchemaVersionDifferences()
    test_instance.test_schema_version_entity_differences()
    test_instance.test_motion_datatype_evolution()
    test_instance.test_schema_version_metadata_differences()
    print("Schema version difference tests completed!")