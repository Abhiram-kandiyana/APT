import unittest
import json
import os
import tempfile
import shutil
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing main
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

from main import APT

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
            
        # Initialize APT
        self.apt = APT(
            system_prompt_1_path=self.sp1_path,
            system_prompt_2_path="dummy", # Not used as file anymore
            selection_method="mdl",
            oracle_path=self.oracle_path,
            debug=False
        )
        # Mock prompt set
        self.apt.prompt_set = []

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("main.vlm_query")
    @patch("subprocess.run")
    def test_oracle_caching(self, mock_subprocess, mock_vlm_query):
        # Setup mocks
        # New flow builds prompt files for all images in this round.
        mock_vlm_query.return_value = [
            (0, "Cached rationale"),
            (1, "Initial VLM rationale")
        ]
        
        # Mock subprocess to simulate tool execution
        # The tool reads input_json and writes output_json
        def side_effect(cmd, check):
            # cmd = [python, oracle.py, --input_file, in, --dataset, ds, --output_file, out]
            input_json_path = cmd[cmd.index("--input_file") + 1]
            output_json_path = cmd[cmd.index("--output_file") + 1]
            
            # Read input to verify what was sent
            with open(input_json_path, 'r') as f:
                data = json.load(f)
            prompts = data["prompts"]
            
            # New integration writes all images of the round.
            self.assertEqual(len(prompts), 2)
            self.assertEqual(prompts[0]["image_path"], self.cached_image_path)
            self.assertEqual(prompts[1]["image_path"], self.new_image_path)
            
            # Write dummy output
            output_data = {"prompts": [
                {
                    "image_path": self.cached_image_path,
                    "class": "wild",
                    "rationale": "Cached rationale corrected"
                },
                {
                    "image_path": self.new_image_path,
                    "class": "lurcher",
                    "rationale": "Corrected rationale",
                    "manual_corrected": True
                }
            ]}
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f)
                
        mock_subprocess.side_effect = side_effect

        # Run oracle_label_and_edit
        images = [self.cached_image_path, self.new_image_path]
        results = self.apt.oracle_label_and_edit(
            images,
            label_map=["wild", "lurcher"],
            dataset="microscopy_lurcher",
            round_num=1
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        # Result 0: From cache
        # wild -> index 0
        self.assertEqual(results[0][0], 0) 
        self.assertIn("Cached rationale corrected", results[0][1])
        
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


    @patch("main.vlm_query")
    @patch("subprocess.run")
    def test_missing_oracle_path_falls_back_to_manual_correction(self, mock_subprocess, mock_vlm_query):
        missing_oracle_path = os.path.join(self.test_dir, "missing_prompt_bank.json")
        fake_tool_path = os.path.join(self.test_dir, "fake_correction_tool.py")
        with open(fake_tool_path, "w") as f:
            f.write("# test placeholder")

        apt = APT(
            system_prompt_1_path=self.sp1_path,
            system_prompt_2_path="dummy",
            selection_method="mdl",
            oracle_path=missing_oracle_path,
            debug=False,
        )
        apt.prompt_set = []

        mock_vlm_query.return_value = [(0, "Initial VLM rationale")]

        def side_effect(cmd, check):
            self.assertEqual(cmd[1], fake_tool_path)
            input_json_path = cmd[cmd.index("--input_json") + 1]
            output_json_path = cmd[cmd.index("--output_json") + 1]
            with open(input_json_path, "r") as f:
                manual_items = json.load(f)
            self.assertEqual(len(manual_items), 1)
            self.assertEqual(manual_items[0]["image_path"], self.new_image_path)
            self.assertEqual(manual_items[0]["label"], "wild")
            corrected = [{
                "image_path": self.new_image_path,
                "label": "lurcher",
                "rationale": "Manual corrected rationale",
            }]
            with open(output_json_path, "w") as f:
                json.dump(corrected, f)

        mock_subprocess.side_effect = side_effect

        result = apt.oracle_label_and_edit(
            self.new_image_path,
            label_map=["wild", "lurcher"],
            dataset="microscopy_lurcher",
            round_num=1,
            correction_tool_path=fake_tool_path,
        )

        self.assertEqual(result[0], 1)
        self.assertIn("Manual corrected rationale", result[1])
        self.assertTrue(os.path.exists(missing_oracle_path))
        with open(missing_oracle_path, "r") as f:
            updated_oracle_data = json.load(f)
        self.assertEqual(len(updated_oracle_data), 1)
        self.assertEqual(updated_oracle_data[0]["class"], "lurcher")
        self.assertEqual(updated_oracle_data[0]["rationale"], "Manual corrected rationale")

if __name__ == "__main__":
    unittest.main()
