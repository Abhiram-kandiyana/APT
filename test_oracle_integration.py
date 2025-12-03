import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import tempfile

# Mock dependencies to avoid environment issues
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import code_mdl
from code_mdl import APTMDL

class TestOracleIntegration(unittest.TestCase):
    @patch('code_mdl.load_files')
    def setUp(self, mock_load_files):
        mock_load_files.return_value = "Dummy System Prompt"
        # Initialize APTMDL with dummy paths
        self.aptmdl = APTMDL(
            system_prompt_1_path="dummy_sp1.md",
            system_prompt_2_path="dummy_sp2.md",
            debug=False # We want to test the real path
        )
        
    @patch('code_mdl.vlm_query')
    @patch('subprocess.run')
    def test_oracle_label_and_edit(self, mock_subprocess, mock_vlm_query):
        # Setup mocks
        
        # 1. Mock VLM Query to return initial predictions
        mock_vlm_query.return_value = [
            (0, "Rationale 1"),
            (1, "Rationale 2")
        ]
        
        # 2. Mock subprocess.run to simulate tool execution
        # We need to intercept the command to get the temp file paths
        # and write the expected output to the output file.
        def side_effect(cmd, check=True):
            # cmd is [python, tool_script, --input_json, in_path, --output_json, out_path, ...]
            self.assertIn("apt_correction_tool_v2.py", cmd[1])
            input_json_path = cmd[3]
            output_json_path = cmd[5]
            
            # Verify input was written correctly
            with open(input_json_path, 'r') as f:
                data = json.load(f)
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0]['label'], 'wild')
                self.assertEqual(data[1]['label'], 'lurcher')
                
            # Simulate user editing: Change label of first image, edit rationale of second
            corrected_data = [
                {
                    "image_path": data[0]['image_path'],
                    "label": "lurcher", # Changed from wild
                    "rationale": "Rationale 1 Edited"
                },
                {
                    "image_path": data[1]['image_path'],
                    "label": "lurcher", # Unchanged
                    "rationale": "Rationale 2 Edited"
                }
            ]
            
            # Write output
            with open(output_json_path, 'w') as f:
                json.dump(corrected_data, f)
                
            return MagicMock()
            
        mock_subprocess.side_effect = side_effect
        
        # Test Data
        images = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
        gen_kwargs = {"label_map": ["wild", "lurcher"]}
        
        # Run method
        results = self.aptmdl.oracle_label_and_edit(images, **gen_kwargs)
        
        # Verify Results
        # Image 1: Changed to lurcher (idx 1)
        self.assertEqual(results[0][0], 1)
        self.assertIn("Rationale 1 Edited", results[0][1])
        self.assertIn("C: lurcher", results[0][1])
        
        # Image 2: Kept lurcher (idx 1)
        self.assertEqual(results[1][0], 1)
        self.assertIn("Rationale 2 Edited", results[1][1])
        self.assertIn("C: lurcher", results[1][1])
        
        # Verify VLM was called
        mock_vlm_query.assert_called_once()
        
        # Verify subprocess was called
        mock_subprocess.assert_called_once()

if __name__ == '__main__':
    unittest.main()
