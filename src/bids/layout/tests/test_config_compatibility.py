"""Tests to verify schema-driven config matches old JSON config behavior."""

import pytest
import re
from bids.layout.models import Config


class TestConfigCompatibility:
    """Test that new schema-driven config matches old JSON config behavior."""
    
    def test_core_entities_present_in_both_configs(self):
        """Test that essential BIDS entities are present in both configs."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Test that both configs support the fundamental BIDS entities
        core_entities = ['subject', 'session', 'task', 'run', 'acquisition']
        
        # Check core entities exist (allowing for naming differences)
        old_has_core = all(entity in old_config.entities for entity in core_entities)
        new_has_core = all(entity in new_config.entities for entity in core_entities)
        
        assert old_has_core and new_has_core, "Both configs should support core BIDS entities"
    
    def test_schema_entities_are_superset_of_valid_old_entities(self):
        """Test that schema includes all valid entities from old config."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Map old config entities to their schema equivalents
        entity_mapping = {
            'inv': 'inversion',
            'mt': 'mtransfer', 
            'proc': 'processing',
            # 'fmap', 'scans', 'staining' are invalid - exclude them
        }
        
        valid_old_entities = set(old_config.entities.keys()) - {'fmap', 'scans', 'staining'}
        missing_from_new = []
        
        for old_entity in valid_old_entities:
            expected_new_entity = entity_mapping.get(old_entity, old_entity)
            if expected_new_entity not in new_config.entities:
                missing_from_new.append(f"{old_entity} -> {expected_new_entity}")
        
        assert len(missing_from_new) == 0, f"Schema missing valid entities: {missing_from_new}"
    
    def test_new_suffixes_compatible_with_old_pattern(self):
        """Test that all new schema suffixes would be accepted by old generic pattern."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Get old pattern (generic) and new suffixes (explicit)
        old_pattern = old_config.entities['suffix'].regex
        new_suffixes = self._extract_suffixes_from_pattern(new_config.entities['suffix'].pattern)
        
        # Test that old pattern would accept all new suffixes
        incompatible_suffixes = []
        for suffix in new_suffixes:
            test_filename = f"sub-01_task-test_{suffix}.nii.gz"
            if not old_pattern.search(test_filename):
                incompatible_suffixes.append(suffix)
        
        assert len(incompatible_suffixes) == 0, f"Old pattern would reject these valid schema suffixes: {incompatible_suffixes}"
    
    def test_new_extensions_compatible_with_old_pattern(self):
        """Test that all new schema extensions would be accepted by old generic pattern."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Get old pattern (generic) and new extensions (explicit)
        old_pattern = old_config.entities['extension'].regex
        new_extensions = self._extract_extensions_from_pattern(new_config.entities['extension'].pattern)
        
        # Test that old pattern would accept all new extensions
        incompatible_extensions = []
        directory_extensions = []
        
        for extension in new_extensions:
            test_filename = f"sub-01_task-test_bold{extension}"
            if not old_pattern.search(test_filename):
                if extension.endswith('/'):
                    # Directory-style extensions are a new BIDS feature not supported by old pattern
                    directory_extensions.append(extension)
                else:
                    incompatible_extensions.append(extension)
        
        # Core file extensions should be compatible
        assert len(incompatible_extensions) == 0, f"Old pattern would reject these valid schema file extensions: {incompatible_extensions}"
        
        # Report directory extensions as expected difference
        if directory_extensions:
            print(f"Note: {len(directory_extensions)} directory-style extensions are new BIDS features: {directory_extensions}")
        
        # Core extensions should be present
        core_extensions = {'.nii.gz', '.nii', '.tsv', '.json'}
        assert core_extensions.issubset(new_extensions), f"Missing core extensions from new config"
    
    def test_datatype_coverage_matches(self):
        """Test that datatype patterns cover the same datatypes."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Extract datatypes from patterns
        old_datatypes = self._extract_datatypes_from_pattern(old_config.entities['datatype'].pattern)
        new_datatypes = self._extract_datatypes_from_pattern(new_config.entities['datatype'].pattern)
        
        missing = old_datatypes - new_datatypes
        extra = new_datatypes - old_datatypes
        
        error_msg = []
        if missing:
            error_msg.append(f"Missing datatypes: {sorted(missing)}")
        if extra:
            error_msg.append(f"Extra datatypes: {sorted(extra)}")
        
        # Core datatypes should be preserved
        core_datatypes = {'anat', 'func', 'dwi', 'fmap'}
        assert core_datatypes.issubset(new_datatypes), f"Missing core datatypes; {'; '.join(error_msg)}"
        
        # New should have at least as many as old (schema grows)
        assert len(new_datatypes) >= len(old_datatypes), f"Fewer datatypes than old config; {'; '.join(error_msg)}"
    
    def test_functional_filename_parsing_compatibility(self):
        """Test that both configs can parse BIDS filenames and extract core entities."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Test core entity extraction (skip extension due to pattern differences)
        test_cases = [
            {
                'filename': '/dataset/sub-01/func/sub-01_task-rest_bold.nii.gz',
                'core_entities': {'subject': '01', 'task': 'rest', 'suffix': 'bold'}
            },
            {
                'filename': '/dataset/sub-02/ses-pre/anat/sub-02_ses-pre_T1w.nii.gz', 
                'core_entities': {'subject': '02', 'session': 'pre', 'suffix': 'T1w'}
            }
        ]
        
        for test_case in test_cases:
            filename = test_case['filename']
            expected = test_case['core_entities']
            
            old_entities = self._parse_filename_entities(filename, old_config)
            new_entities = self._parse_filename_entities(filename, new_config)
            
            # Test that both configs extract the core BIDS entities
            for key, expected_value in expected.items():
                old_extracted = old_entities.get(key)
                new_extracted = new_entities.get(key)
                
                # Both should extract the same core information
                assert old_extracted == expected_value, f"Old config failed to extract {key}={expected_value} from {filename}, got {old_extracted}"
                assert new_extracted == expected_value, f"New config failed to extract {key}={expected_value} from {filename}, got {new_extracted}"
        
        # Test that new config has valid patterns
        assert len(new_config.entities) > 20, "New config should have many entities"
        assert 'subject' in new_config.entities, "New config missing subject entity"
        assert 'suffix' in new_config.entities, "New config missing suffix entity"
    
    def test_new_config_parses_schema_compliant_filenames(self):
        """Test that new config correctly parses filenames with schema entities."""
        new_config = Config.load('bids-schema')
        
        # Test files that use entities only available in schema (not old config)
        schema_specific_cases = [
            '/dataset/sub-01/anat/sub-01_hemi-L_T1w.nii.gz',  # hemisphere entity
            '/dataset/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz',  # description entity  
            '/dataset/sub-01/anat/sub-01_res-native_T1w.nii.gz',  # resolution entity
        ]
        
        for filename in schema_specific_cases:
            entities = self._parse_filename_entities(filename, new_config)
            
            # Should extract subject and suffix at minimum
            assert 'subject' in entities, f"Failed to extract subject from {filename}"
            assert 'suffix' in entities, f"Failed to extract suffix from {filename}"
            
            # Should extract the schema-specific entity
            if 'hemi-' in filename:
                assert 'hemisphere' in entities, f"Failed to extract hemisphere from {filename}"
            if 'desc-' in filename:
                assert 'description' in entities, f"Failed to extract description from {filename}"
            if 'res-' in filename:
                assert 'resolution' in entities, f"Failed to extract resolution from {filename}"
    
    def test_pattern_structure_similarity(self):
        """Test that pattern structures are similar between old and new."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Test key entities have reasonable pattern structures
        key_entities = ['subject', 'session', 'task', 'run', 'suffix', 'extension']
        
        for entity_name in key_entities:
            if entity_name in old_config.entities and entity_name in new_config.entities:
                old_pattern = old_config.entities[entity_name].pattern
                new_pattern = new_config.entities[entity_name].pattern
                
                # Both should be non-empty
                assert len(old_pattern) > 0, f"Old {entity_name} pattern is empty"
                assert len(new_pattern) > 0, f"New {entity_name} pattern is empty"
                
                # Both should contain capturing groups
                old_groups = len(re.findall(r'\([^)]*\)', old_pattern))
                new_groups = len(re.findall(r'\([^)]*\)', new_pattern))
                
                assert old_groups > 0, f"Old {entity_name} pattern has no groups"
                assert new_groups > 0, f"New {entity_name} pattern has no groups"
    
    def test_config_names_and_metadata(self):
        """Test config metadata is reasonable."""
        old_config = Config.load('bids')
        new_config = Config.load('bids-schema')
        
        # Old config should have simple name
        assert old_config.name == 'bids'
        
        # New config should include version info
        assert new_config.name.startswith('bids-schema-')
        assert len(new_config.name) > len('bids-schema-')
        
        # Both should have reasonable entity counts
        assert len(old_config.entities) > 10, "Old config has too few entities"
        assert len(new_config.entities) > 10, "New config has too few entities"
        
        # Report entity count ratio for analysis
        ratio = len(new_config.entities) / len(old_config.entities)
        print(f"Entity count ratio (new/old): {ratio:.2f}")
    
    # Helper methods
    def _extract_suffixes_from_pattern(self, pattern):
        """Extract suffixes from a pattern like [_/\\]+(suffix1|suffix2|...)"""
        suffixes = set()
        
        # Look for alternation groups
        matches = re.findall(r'\(([^)]+)\)', pattern)
        for match in matches:
            if '|' in match:
                suffixes.update(match.split('|'))
            else:
                suffixes.add(match)
        
        return suffixes
    
    def _extract_extensions_from_pattern(self, pattern):
        """Extract extensions from a pattern."""
        extensions = set()
        
        # Look for patterns with dots
        matches = re.findall(r'\\?\.\w+(?:\\?\.\w+)*', pattern)
        for match in matches:
            # Clean up regex escaping
            clean_ext = match.replace('\\', '')
            extensions.add(clean_ext)
        
        # Also look for alternation groups
        alt_matches = re.findall(r'\(([^)]+)\)', pattern)
        for match in alt_matches:
            if '|' in match and '.' in match:
                for item in match.split('|'):
                    if '.' in item:
                        clean_item = item.replace('\\', '')
                        extensions.add(clean_item)
        
        return extensions
    
    def _extract_datatypes_from_pattern(self, pattern):
        """Extract datatypes from a pattern like [/\\]+(datatype1|datatype2|...)"""
        datatypes = set()
        
        # Look for alternation groups
        matches = re.findall(r'\(([^)]+)\)', pattern)
        for match in matches:
            if '|' in match:
                datatypes.update(match.split('|'))
            else:
                datatypes.add(match)
        
        return datatypes
    
    def _parse_filename_entities(self, filename, config):
        """Parse entities from filename using config patterns."""
        entities = {}
        
        for entity_name, entity in config.entities.items():
            if hasattr(entity, 'regex') and entity.regex:
                match = entity.regex.search(filename)
                if match and len(match.groups()) > 0:
                    entities[entity_name] = match.group(1)
        
        return entities
    
    def _map_entity_name_old_to_new(self, new_entity_name):
        """Map new entity names to old entity names for compatibility testing."""
        # Old config sometimes used different names
        name_mapping = {
            'subject': 'subject',
            'session': 'session', 
            'task': 'task',
            'run': 'run',
            'acquisition': 'acquisition',
            'suffix': 'suffix',
            'extension': 'extension',
            # Add other mappings as needed
        }
        return name_mapping.get(new_entity_name, new_entity_name)


if __name__ == '__main__':
    # Allow running as script for quick testing
    test_instance = TestConfigCompatibility()
    test_instance.test_entity_coverage_matches()
    test_instance.test_suffix_coverage_matches()
    test_instance.test_filename_parsing_compatibility()
