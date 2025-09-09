"""Tests to verify schema patterns match actual schema content."""

from bids.layout.models import Config
from bidsschematools import schema as bst_schema


class TestSchemaAccuracy:
    """Test that patterns accurately reflect BIDS schema content."""
    
    def test_extension_patterns_match_schema(self):
        """Test that extension patterns match schema extensions."""
        config = Config.load('bids-schema')
        
        # Load schema directly
        bids_schema = bst_schema.load_schema()
        
        # Get extension entity
        extension_entity = config.entities['extension']
        
        # Test that schema extensions are matched (excluding wildcards)
        schema_extensions = [ext['value'] for ext in bids_schema.objects.extensions.values() 
                           if ext.get('value') and ext['value'].startswith('.') and ext['value'] not in ('.*',)]
        
        # Collect all failures
        matched_extensions = []
        unmatched_extensions = []
        
        test_files = [f'test{ext}' for ext in schema_extensions]
        for ext, test_file in zip(schema_extensions, test_files):
            match = extension_entity.regex.search(test_file)
            if match is not None:
                matched_extensions.append(ext)
            else:
                unmatched_extensions.append(ext)
        
        # Report all differences at once
        coverage = len(matched_extensions) / len(schema_extensions) * 100 if schema_extensions else 0
        
        error_msg = []
        error_msg.append(f"Extension coverage: {coverage:.1f}% ({len(matched_extensions)}/{len(schema_extensions)})")
        if unmatched_extensions:
            error_msg.append(f"Unmatched extensions ({len(unmatched_extensions)}): {unmatched_extensions[:20]}")
            if len(unmatched_extensions) > 20:
                error_msg.append(f"... and {len(unmatched_extensions) - 20} more")
        if matched_extensions:
            error_msg.append(f"Matched extensions ({len(matched_extensions)}): {matched_extensions[:10]}")
            if len(matched_extensions) > 10:
                error_msg.append(f"... and {len(matched_extensions) - 10} more")
        
        assert len(unmatched_extensions) == 0, "; ".join(error_msg)
    
    def test_suffix_patterns_match_schema(self):
        """Test that suffix patterns match schema suffixes."""
        config = Config.load('bids-schema')
        
        # Load schema directly
        bids_schema = bst_schema.load_schema()
        
        # Get suffix entity
        suffix_entity = config.entities['suffix']
        
        # Test that schema suffixes are matched using actual suffix values, not keys
        schema_suffix_values = [obj['value'] for obj in bids_schema.objects.suffixes.values() 
                               if obj.get('value')]
        
        # Collect all failures
        matched_suffixes = []
        unmatched_suffixes = []
        
        # Test all suffixes using their actual values
        test_files = [f'_{suffix}.nii.gz' for suffix in schema_suffix_values]
        for suffix, test_file in zip(schema_suffix_values, test_files):
            match = suffix_entity.regex.search(test_file)
            if match is not None:
                matched_suffixes.append(suffix)
            else:
                unmatched_suffixes.append(suffix)
        
        # Report all differences at once
        coverage = len(matched_suffixes) / len(schema_suffix_values) * 100 if schema_suffix_values else 0
        
        error_msg = []
        error_msg.append(f"Suffix coverage: {coverage:.1f}% ({len(matched_suffixes)}/{len(schema_suffix_values)})")
        if unmatched_suffixes:
            error_msg.append(f"Unmatched suffixes ({len(unmatched_suffixes)}): {unmatched_suffixes[:20]}")
            if len(unmatched_suffixes) > 20:
                error_msg.append(f"... and {len(unmatched_suffixes) - 20} more")
        if matched_suffixes:
            error_msg.append(f"Matched suffixes ({len(matched_suffixes)}): {matched_suffixes[:10]}")
            if len(matched_suffixes) > 10:
                error_msg.append(f"... and {len(matched_suffixes) - 10} more")
        
        assert len(unmatched_suffixes) == 0, "; ".join(error_msg)
    
    def test_datatype_patterns_match_schema(self):
        """Test that datatype patterns match schema datatypes."""
        config = Config.load('bids-schema')
        
        # Load schema directly
        bids_schema = bst_schema.load_schema()
        
        # Get datatype entity
        datatype_entity = config.entities['datatype']
        
        # Test that schema datatypes are matched
        schema_datatypes = list(bids_schema.objects.datatypes.keys())
        
        # Collect all failures
        matched_datatypes = []
        unmatched_datatypes = []
        
        test_paths = [f'/{dtype}/' for dtype in schema_datatypes]
        for dtype, test_path in zip(schema_datatypes, test_paths):
            match = datatype_entity.regex.search(test_path)
            if match is not None:
                matched_datatypes.append(dtype)
            else:
                unmatched_datatypes.append(dtype)
        
        # Report all differences at once
        coverage = len(matched_datatypes) / len(schema_datatypes) * 100 if schema_datatypes else 0
        
        error_msg = []
        error_msg.append(f"Datatype coverage: {coverage:.1f}% ({len(matched_datatypes)}/{len(schema_datatypes)})")
        if unmatched_datatypes:
            error_msg.append(f"Unmatched datatypes ({len(unmatched_datatypes)}): {unmatched_datatypes}")
        if matched_datatypes:
            error_msg.append(f"Matched datatypes ({len(matched_datatypes)}): {matched_datatypes}")
        
        assert len(unmatched_datatypes) == 0, "; ".join(error_msg)
    
    def test_entity_format_patterns_match_schema(self):
        """Test that entity patterns use actual schema format patterns."""
        config = Config.load('bids-schema')
        
        # Load schema directly
        bids_schema = bst_schema.load_schema()
        
        # Test a few key entities
        test_entities = [
            ('subject', 'label'),
            ('run', 'index'),
            ('echo', 'index'),
        ]
        
        for entity_name, expected_format in test_entities:
            if entity_name in config.entities:
                entity = config.entities[entity_name]
                
                # Get expected pattern from schema
                format_obj = bids_schema.objects.formats[expected_format]
                expected_pattern = format_obj['pattern']
                
                # Check that our entity pattern contains the schema pattern
                assert expected_pattern in entity.pattern, \
                    f"Entity {entity_name} pattern doesn't contain schema format pattern"
    
    def test_no_hardcoded_values(self):
        """Test that we don't have hardcoded schema values anymore."""
        config = Config.load('bids-schema')
        
        # Load schema directly to get actual counts
        bids_schema = bst_schema.load_schema()
        
        # We should have more entities than before because we're not missing any
        assert len(config.entities) >= len(bids_schema.objects.entities)
        
        # Extension pattern should include schema extensions
        extension_entity = config.entities['extension']
        schema_extension_count = len([ext for ext in bids_schema.objects.extensions.values() 
                                    if ext.get('value') and ext['value'].startswith('.')])
        
        # Pattern should include many extensions (not just a generic catch-all)
        assert len(extension_entity.pattern) > 100, "Extension pattern seems too simple"
        
        # Suffix pattern should include schema suffixes
        suffix_entity = config.entities['suffix']
        schema_suffix_count = len(bids_schema.objects.suffixes)
        
        # Pattern should include many suffixes
        assert len(suffix_entity.pattern) > 200, "Suffix pattern seems too simple"
        
        print(f"Schema extensions used: {schema_extension_count}")
        print(f"Schema suffixes used: {schema_suffix_count}")
        print(f"Schema datatypes used: {len(bids_schema.objects.datatypes)}")
        print(f"Schema entities used: {len(bids_schema.objects.entities)}")


if __name__ == '__main__':
    # Allow running as script for quick testing
    test_instance = TestSchemaAccuracy()
    test_instance.test_extension_patterns_match_schema()
    test_instance.test_suffix_patterns_match_schema()
    test_instance.test_datatype_patterns_match_schema()
    test_instance.test_entity_format_patterns_match_schema()
    test_instance.test_no_hardcoded_values()
    print("All schema accuracy tests passed!")