import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock sentence_transformers before importing main to avoid loading heavy models/dependencies
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers'].SentenceTransformer = MagicMock()

from main import vlm_query, parse_vlm_response, APT, mdl_loss

class TestBatchProcessing(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_choice = MagicMock()
        self.mock_message = MagicMock()
        
        self.mock_response.choices = [self.mock_choice]
        self.mock_choice.message = self.mock_message
        
        # Patch client in main
        patcher = patch('main.client', self.mock_client)
        self.mock_client_patch = patcher.start()
        self.mock_client.chat.completions.create.return_value = self.mock_response
        self.addCleanup(patcher.stop)
        
        # Patch encode_image to avoid file IO
        patcher_enc = patch('main.encode_image', return_value="base64string")
        self.mock_encode = patcher_enc.start()
        self.addCleanup(patcher_enc.stop)
        
        # Patch text_encoder to avoid loading model
        patcher_te = patch('main.text_encoder', return_value=[[0.1]*384])
        self.mock_te = patcher_te.start()
        self.addCleanup(patcher_te.stop)

    def test_parse_vlm_response_batch(self):
        output_text = """
        R: Rationale 1 C: wild
        R: Rationale 2 C: lurcher
        """
        gen_kwargs = {"label_map": ["wild", "lurcher"]}
        results = parse_vlm_response(output_text, gen_kwargs)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], (0, "Rationale 1"))
        self.assertEqual(results[1], (1, "Rationale 2"))

    def test_vlm_query_batch_success(self):
        self.mock_message.content = "R: R1 C: wild R: R2 C: lurcher"
        
        images = ["img1.jpg", "img2.jpg"]
        sp1 = "System Prompt 1"
        sp2_template = "Classify {N} images"
        prompt_set = []
        
        results = vlm_query(
            images, sp1, sp2_template, prompt_set,
            label_map=["wild", "lurcher"],
            vlm_log_path="test_log.json"
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[1][0], 1)
        
        # Verify prompt formatting
        call_args = self.mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        content = messages[0]['content']
        
        # Check if sp2 was formatted correctly
        sp2_found = False
        for item in content:
            if item.get('text') == "Classify 2 images":
                sp2_found = True
                break
        self.assertTrue(sp2_found)

    def test_vlm_query_mismatch_retries_then_counts_invalid(self):
        self.mock_message.content = "R: R1 C: wild"
        
        images = ["img1.jpg", "img2.jpg"]
        sp1 = "System Prompt 1"
        sp2_template = "Classify {N} images"
        prompt_set = []
        invalid_output_stats = {}
        
        results = vlm_query(
            images, sp1, sp2_template, prompt_set,
            label_map=["wild", "lurcher"],
            invalid_output_stats=invalid_output_stats,
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], -1)
        self.assertEqual(results[1][0], -1)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
        self.assertEqual(invalid_output_stats["prediction_count_mismatch"], 3)
        self.assertEqual(invalid_output_stats["exhausted_invalid_output_retries"], 2)

    def test_vlm_query_invalid_label_retries_then_counts_invalid(self):
        self.mock_message.content = "R: R1 C: garbage"
        invalid_output_stats = {}

        results = vlm_query(
            ["img1.jpg"],
            "System Prompt 1",
            "Classify {N} images",
            [],
            label_map=["wild", "lurcher"],
            invalid_output_stats=invalid_output_stats,
        )

        self.assertEqual(results[0][0], -1)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
        self.assertEqual(invalid_output_stats["invalid_label"], 3)
        self.assertEqual(invalid_output_stats["exhausted_invalid_output_retries"], 1)

    def test_vlm_query_invalid_label_respects_single_attempt_setting(self):
        self.mock_message.content = "R: R1 C: garbage"
        invalid_output_stats = {}

        results = vlm_query(
            ["img1.jpg"],
            "System Prompt 1",
            "Classify {N} images",
            [],
            label_map=["wild", "lurcher"],
            invalid_output_stats=invalid_output_stats,
            invalid_output_max_retries=1,
        )

        self.assertEqual(results[0][0], -1)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 1)
        self.assertEqual(invalid_output_stats["invalid_label"], 1)
        self.assertEqual(invalid_output_stats["exhausted_invalid_output_retries"], 1)

    def test_vlm_query_invalid_label_can_recover_on_retry(self):
        responses = ["R: R1 C: garbage", "R: R1 C: wild"]

        def side_effect(*args, **kwargs):
            self.mock_message.content = responses[self.mock_client.chat.completions.create.call_count - 1]
            return self.mock_response

        self.mock_client.chat.completions.create.side_effect = side_effect

        results = vlm_query(
            ["img1.jpg"],
            "System Prompt 1",
            "Classify {N} images",
            [],
            label_map=["wild", "lurcher"]
        )

        self.assertEqual(results[0][0], 0)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)

    def test_mdl_loss_batching(self):
        # Mock vlm_query to avoid complexity
        with patch('main.vlm_query') as mock_vlm:
            mock_vlm.return_value = [(0, "R1"), (1, "R2")]
            
            val_data = [("img1.jpg", 0), ("img2.jpg", 1), ("img3.jpg", 0)]
            prompt_set = []
            sp1 = "SP1"
            sp2_template = "SP2 {N}"
            
            # Mock description_length to return 0
            with patch('main.description_length', return_value=0.0):
                loss = mdl_loss(
                    prompt_set, val_data, sp1, sp2_template,
                    alpha=0.01, beta=0.1, lambda_mdl=0.1,
                    val_batch_size=2, # Batch size 2, so 2 calls (2 items, then 1 item)
                    label_map=["wild", "lurcher"]
                )
                
            # Verify vlm_query was called twice
            self.assertEqual(mock_vlm.call_count, 2)
            
            # First call with 2 images
            args1, _ = mock_vlm.call_args_list[0]
            self.assertEqual(len(args1[0]), 2)
            
            # Second call with 1 image
            args2, _ = mock_vlm.call_args_list[1]
            self.assertEqual(len(args2[0]), 1)

if __name__ == '__main__':
    unittest.main()
