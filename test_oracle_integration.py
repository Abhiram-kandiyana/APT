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

# Now import main
from main import APT

class TestOracleIntegration(unittest.TestCase):
    @patch('main.load_files')
    def setUp(self, mock_load_files):
        mock_load_files.return_value = "Dummy System Prompt"
        # Initialize APT with dummy paths
        self.apt = APT(
            system_prompt_1_path="dummy_sp1.md",
            system_prompt_2_path="dummy_sp2.md",
            selection_method="mdl",
            debug=False # We want to test the real path
        )
        
    @patch('main.vlm_query')
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
            self.assertTrue(cmd[1].endswith("oracle.py"))
            input_json_path = cmd[cmd.index("--input_file") + 1]
            output_json_path = cmd[cmd.index("--output_file") + 1]
            
            # Verify input was written correctly
            with open(input_json_path, 'r') as f:
                data = json.load(f)
                prompts = data["prompts"]
                self.assertEqual(len(prompts), 2)
                self.assertEqual(prompts[0]['class'], 'wild')
                self.assertEqual(prompts[1]['class'], 'lurcher')
                
            # Simulate user editing: Change label of first image, edit rationale of second
            corrected_data = {"prompts": [
                {
                    "image_path": prompts[0]['image_path'],
                    "class": "lurcher", # Changed from wild
                    "rationale": "Rationale 1 Edited"
                },
                {
                    "image_path": prompts[1]['image_path'],
                    "class": "lurcher", # Unchanged
                    "rationale": "Rationale 2 Edited"
                }
            ]}
            
            # Write output
            with open(output_json_path, 'w') as f:
                json.dump(corrected_data, f)
                
            return MagicMock()
            
        mock_subprocess.side_effect = side_effect
        
        # Test Data
        images = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
        gen_kwargs = {"label_map": ["wild", "lurcher"], "dataset": "microscopy_lurcher", "round_num": 1}
        
        # Run method
        results = self.apt.oracle_label_and_edit(images, **gen_kwargs)
        
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
