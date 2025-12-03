import unittest
import json
import os
import tempfile
import shutil
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing code_mdl
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

from code_mdl import APTMDL

class TestOracleCaching(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a dummy oracle.json
        self.oracle_path = os.path.join(self.test_dir, "oracle.json")
        self.cached_image_path = os.path.abspath(os.path.join(self.test_dir, "cached_image.jpg"))
        # Create dummy image file
        with open(self.cached_image_path, 'w') as f:
            f.write("dummy image content")
            
        self.cached_data = [
            {
                "image_path": self.cached_image_path,
                "class": "wild",
                "rationale": "Cached rationale"
            }
        ]
        with open(self.oracle_path, 'w') as f:
            json.dump(self.cached_data, f)
            
        # Create a non-cached image
        self.new_image_path = os.path.abspath(os.path.join(self.test_dir, "new_image.jpg"))
        with open(self.new_image_path, 'w') as f:
            f.write("dummy new image content")

        # Create dummy system prompts
        self.sp1_path = os.path.join(self.test_dir, "sp1.md")
        with open(self.sp1_path, 'w') as f:
            f.write("System Prompt 1")
            
        # Initialize APTMDL
        self.aptmdl = APTMDL(
            system_prompt_1_path=self.sp1_path,
            system_prompt_2_path="dummy", # Not used as file anymore
            oracle_path=self.oracle_path,
            debug=False
        )
        # Mock prompt set
        self.aptmdl.prompt_set = []

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("code_mdl.vlm_query")
    @patch("subprocess.run")
    def test_oracle_caching(self, mock_subprocess, mock_vlm_query):
        # Setup mocks
        # Mock VLM query to return a prediction for the new image
        mock_vlm_query.return_value = [(1, "Initial VLM rationale")] # 1 -> lurcher
        
        # Mock subprocess to simulate tool execution
        # The tool reads input_json and writes output_json
        def side_effect(cmd, check):
            # Extract input and output paths from cmd
            # cmd = ["python", "apt_correction_tool_v2.py", "--input_json", tmp_in, "--output_json", tmp_out, ...]
            input_json_path = cmd[3]
            output_json_path = cmd[5]
            
            # Read input to verify what was sent
            with open(input_json_path, 'r') as f:
                data = json.load(f)
            
            # Should only contain the new image
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["image_path"], self.new_image_path)
            
            # Write dummy output
            output_data = [
                {
                    "image_path": self.new_image_path,
                    "label": "lurcher",
                    "rationale": "Corrected rationale"
                }
            ]
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f)
                
        mock_subprocess.side_effect = side_effect

        # Run oracle_label_and_edit
        images = [self.cached_image_path, self.new_image_path]
        results = self.aptmdl.oracle_label_and_edit(images, label_map=["wild", "lurcher"])
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        # Result 0: From cache
        # wild -> index 0
        self.assertEqual(results[0][0], 0) 
        self.assertIn("Cached rationale", results[0][1])
        
        # Result 1: From tool
        # lurcher -> index 1
        self.assertEqual(results[1][0], 1)
        self.assertIn("Corrected rationale", results[1][1])
        
        # Verify oracle.json was updated
        with open(self.oracle_path, 'r') as f:
            updated_oracle_data = json.load(f)
            
        self.assertEqual(len(updated_oracle_data), 2)
        # Check if new entry is present
        new_entry = next((item for item in updated_oracle_data if item["image_path"] == self.new_image_path), None)
        self.assertIsNotNone(new_entry)
        self.assertEqual(new_entry["class"], "lurcher")
        self.assertEqual(new_entry["rationale"], "Corrected rationale")

if __name__ == "__main__":
    unittest.main()
