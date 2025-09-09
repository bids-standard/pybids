"""Tests for schema-based Config loading."""

import pytest

from bids.layout.models import Config


class TestSchemaConfig:
    """Test schema-based configuration loading."""
    
    def test_load_bids_schema_basic(self):
        """Test loading config from BIDS schema."""
        config = Config.load('bids-schema')
        
        # Basic checks
        assert config.name.startswith('bids-schema-')
        assert len(config.entities) > 0
        
        # Check that standard entities are present (using full names per PyBIDS convention)
        entity_names = {e.name for e in config.entities.values()}
        expected_entities = {'subject', 'session', 'task', 'run'}
        assert expected_entities.issubset(entity_names)
        
        # Check that PyBIDS-specific entities are present
        pybids_entities = {'extension', 'suffix', 'datatype'}
        assert pybids_entities.issubset(entity_names)
    
    def test_schema_entity_patterns(self):
        """Test that entity patterns are correctly generated."""
        config = Config.load('bids-schema')
        
        # Test subject entity (directory-based)
        if 'subject' in config.entities:
            subject_entity = config.entities['subject']
            assert subject_entity.regex is not None
            # Should match /sub-01/ or similar
            match = subject_entity.regex.search('/sub-01/')
            assert match is not None
            assert match.group(1) == '01'
        
        # Test task entity (file-based)
        if 'task' in config.entities:
            task_entity = config.entities['task']
            assert task_entity.regex is not None
            # Should match _task-rest_ or similar
            match = task_entity.regex.search('_task-rest_')
            assert match is not None
            assert match.group(1) == 'rest'
    
    def test_schema_version_tracking(self):
        """Test that schema version is tracked in config name."""
        config = Config.load('bids-schema')
        
        # Config name should include version
        assert 'bids-schema-' in config.name
        # Should have some version number after the dash
        version_part = config.name.split('bids-schema-')[1]
        assert len(version_part) > 0
    
    def test_schema_version_parameter(self):
        """Test loading specific schema version (future feature)."""
        # This should work but use the default schema for now
        config = Config.load({'schema_version': '1.9.0'})
        
        assert config.name.startswith('bids-schema-')
        assert len(config.entities) > 0
    
    
    def test_entity_generation(self):
        """Test entity pattern generation using real schema."""
        config = Config.load('bids-schema')
        
        # Test that entities were generated
        assert len(config.entities) > 0
        
        # Test specific entity properties we care about (using full names)
        assert 'subject' in config.entities
        assert 'task' in config.entities
        assert 'run' in config.entities
        
        # Test that patterns are valid regex
        for entity_name, entity in config.entities.items():
            assert entity.regex is not None, f"Entity {entity_name} missing regex"
            # Test that regex compiles and works
            assert hasattr(entity.regex, 'search'), f"Entity {entity_name} regex invalid"
        
        # Test specific entity patterns work correctly
        subject_entity = config.entities['subject']
        assert 'sub-' in subject_entity.pattern
        
        # Test subject pattern matches expected format
        match = subject_entity.regex.search('/sub-01/')
        assert match is not None
        assert match.group(1) == '01'
        
        # Test task pattern matches expected format  
        task_entity = config.entities['task']
        match = task_entity.regex.search('_task-rest_')
        assert match is not None
        assert match.group(1) == 'rest'
        
        # Test run pattern matches expected format
        run_entity = config.entities['run']
        match = run_entity.regex.search('_run-02_')
        assert match is not None
        assert match.group(1) == '02'
    


if __name__ == '__main__':
    # Allow running as script for quick testing
    test_instance = TestSchemaConfig()
    test_instance.test_load_bids_schema_basic()
    test_instance.test_schema_entity_patterns()
    test_instance.test_schema_version_tracking()
    print("Basic tests passed!")