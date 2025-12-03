import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json

# Add parent directory to path to import code_mdl
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock sentence_transformers before importing code_mdl to avoid loading heavy models/dependencies
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers'].SentenceTransformer = MagicMock()

from code_mdl import vlm_query, parse_vlm_response, APTMDL, mdl_loss

class TestBatchProcessing(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_choice = MagicMock()
        self.mock_message = MagicMock()
        
        self.mock_response.choices = [self.mock_choice]
        self.mock_choice.message = self.mock_message
        
        # Patch client in code_mdl
        patcher = patch('code_mdl.client', self.mock_client)
        self.mock_client_patch = patcher.start()
        self.mock_client.chat.completions.create.return_value = self.mock_response
        self.addCleanup(patcher.stop)
        
        # Patch encode_image to avoid file IO
        patcher_enc = patch('code_mdl.encode_image', return_value="base64string")
        self.mock_encode = patcher_enc.start()
        self.addCleanup(patcher_enc.stop)
        
        # Patch text_encoder to avoid loading model
        patcher_te = patch('code_mdl.text_encoder', return_value=[[0.1]*384])
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

    def test_vlm_query_retry_logic(self):
        # First call returns 1 result (mismatch), second call returns 2 (success)
        self.mock_message.content = "R: R1 C: wild"
        
        # Side effect to change return value
        def side_effect(*args, **kwargs):
            if self.mock_client.chat.completions.create.call_count == 1:
                self.mock_message.content = "R: R1 C: wild"
            else:
                self.mock_message.content = "R: R1 C: wild R: R2 C: lurcher"
            return self.mock_response
            
        self.mock_client.chat.completions.create.side_effect = side_effect
        
        images = ["img1.jpg", "img2.jpg"]
        sp1 = "System Prompt 1"
        sp2_template = "Classify {N} images"
        prompt_set = []
        
        results = vlm_query(
            images, sp1, sp2_template, prompt_set,
            label_map=["wild", "lurcher"]
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)

    def test_mdl_loss_batching(self):
        # Mock vlm_query to avoid complexity
        with patch('code_mdl.vlm_query') as mock_vlm:
            mock_vlm.return_value = [(0, "R1"), (1, "R2")]
            
            val_data = [("img1.jpg", 0), ("img2.jpg", 1), ("img3.jpg", 0)]
            prompt_set = []
            sp1 = "SP1"
            sp2_template = "SP2 {N}"
            
            # Mock description_length to return 0
            with patch('code_mdl.description_length', return_value=0.0):
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
