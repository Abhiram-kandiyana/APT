import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np
import shutil
import tempfile

# Mock external dependencies before importing main to avoid environment issues
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

# Add current directory to path
sys.path.append(os.getcwd())

from main import selection_score, caption_complexity, uncertainty, expected_caption_complexity

class TestSingleSelection(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for dummy images
        self.test_dir = tempfile.mkdtemp()
        self.images = []
        for i in range(2):
            img_path = os.path.join(self.test_dir, f"img_{i}.jpg")
            with open(img_path, 'w') as f:
                f.write("dummy image content")
            self.images.append(img_path)
            
        self.prompt_set = [("p1.jpg", "caption 1"), ("p2.jpg", "caption 2")]
        self.prompt_embeddings = np.random.rand(2, 384)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('main.vlm_query')
    @patch('main.text_encoder')
    @patch('main.tokenizer')
    def test_single_selection_score(self, mock_tokenizer, mock_text_encoder, mock_vlm_query):
        # Setup mocks
        mock_tokenizer.side_effect = lambda x: [1] * len(x) # length 1 for simplicity
        
        # Mock text_encoder to return deterministic embeddings based on caption content
        def text_encoder_side_effect(captions):
            embeddings = []
            for cap in captions:
                # Simple hash-like generation
                seed = sum(ord(c) for c in cap) % 2**32
                rng = np.random.RandomState(seed)
                embeddings.append(rng.rand(384))
            return np.array(embeddings)
            
        mock_text_encoder.side_effect = text_encoder_side_effect
        
        # Mock vlm_query to return consistent results based on input
        # vlm_query returns (label, rationale) or list of them
        def vlm_side_effect(x, *args, **kwargs):
            is_batch = isinstance(x, list)
            imgs = x if is_batch else [x]
            results = []
            for img in imgs:
                # Deterministic result based on image name
                try:
                    idx = int(os.path.basename(img).split('_')[1].split('.')[0])
                except:
                    idx = 0
                label = idx % 2
                rationale = f"Rationale for {idx}"
                results.append((label, rationale))
            
            if is_batch:
                return results
            else:
                return results[0]
                
        mock_vlm_query.side_effect = vlm_side_effect
        
        # Verify call with single image
        img = self.images[0]
        score = selection_score(
            img, "s1", "s2", self.prompt_set, self.prompt_embeddings,
            lambda_c=0.5, K=2, alpha=0.01, beta=0.1,
            label_map=["lurcher", "wild"]
        )
        
        print(f"Single item score: {score}")
        self.assertIsInstance(score, float)
        
        # Verify VLM calls were made correctly (not batched inside, but still works)
        # selection_score calls:
        # 1. uncertainty -> loop K=2 times -> K calls to vlm_query (for single item)
        # 2. expected_caption_complexity -> 1 call to vlm_query (stochastic=False)
        # Total 3 calls expected for single item
        self.assertEqual(mock_vlm_query.call_count, 3)
        
        # Verify first call args (uncertainty)
        args, _ = mock_vlm_query.call_args_list[0]
        # x passed to vlm_query should be the image string or whatever (here path string)
        # In uncertainty loop: vlm_query(x, ...)
        
        # Wait, if uncertainty is called with a single item `x` (path string),
        # inside uncertainty:
        # is_batch = False
        # ...
        # vlm_query(x, ...) -> vlm_query receives string.
        # But wait, lines 507 append to batch_to_query.
        # Let's check `uncertainty` implementation in main.py for single item path.
        
        # In `uncertainty` (now modified to handle both):
        # is_batch = False (since x is string)
        # images = [x]
        # loop K times:
        #   batch_to_query = []
        #   loop over images (length 1):
        #     batch_to_query.append(img)
        #   vlm_query(batch_to_query, ...)
        # So vlm_query is called with a LIST of length 1 even for single item input,
        # because I changed uncertainty to support batching internally and always use list for `vlm_query`?
        # Let's check lines 498-515 in main.py.
        # Yes: `batch_to_query.append(img)` then `vlm_query(batch_to_query, ...)`

        # So vlm_query receives a list [img_path].

        passed_arg = args[0]
        self.assertIsInstance(passed_arg, list)
        self.assertEqual(len(passed_arg), 1)
        self.assertEqual(passed_arg[0], img)

if __name__ == '__main__':
    unittest.main()
