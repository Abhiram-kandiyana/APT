import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os

# Mock dependencies before importing anything that uses them
sys.modules['openai'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

# Add current directory to path to import main
sys.path.append(os.getcwd())

import main

class TestVLMQuery(unittest.TestCase):
    @patch('main.client')
    @patch('main.encode_image')
    def test_vlm_query_success(self, mock_encode, mock_client):
        # Setup mocks
        mock_encode.return_value = "base64string"
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "'e': This is a test explanation. 'c': T"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test data
        x = "dummy_path.jpg"
        sp1_path = "sp1.md"
        sp2_path = "sp2.md"
        prompt_set = [("ex1.jpg", "caption1")]
        
        # Mock file reading
        file_content = {sp1_path: "System Prompt 1 Content", sp2_path: "System Prompt 2 Content"}
        
        def side_effect(filename, *args, **kwargs):
            if filename in file_content:
                return mock_open(read_data=file_content[filename])()
            return mock_open()()

        with patch('builtins.open', side_effect=side_effect):
            # Call function
            label, explanation = main.vlm_query(x, sp1_path, sp2_path, prompt_set)
        
        # Assertions
        self.assertEqual(label, 1) # T -> 1 (fallback)
        self.assertEqual(explanation, "This is a test explanation.")

    @patch('main.client')
    @patch('main.encode_image')
    def test_vlm_query_dynamic_mapping(self, mock_encode, mock_client):
        mock_encode.return_value = "base64string"
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "'e': Explanation 'c': L"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data="content")):
             # Provide label_map: W->0, L->1
             label, explanation = main.vlm_query("x", "p1", "p2", [], label_map=['W', 'L'])
             
        self.assertEqual(label, 1) # L -> 1
        self.assertEqual(explanation, "Explanation")

    def test_text_encoder(self):
        # Mock the st_model.encode method
        main.st_model = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.__len__.return_value = 1
        main.st_model.encode.return_value = mock_embeddings
        
        captions = ["test caption"]
        embeddings = main.text_encoder(captions)
        
        main.st_model.encode.assert_called_with(captions)
        self.assertTrue(len(embeddings) > 0)

    def test_tokenizer(self):
        text = "Hello, world!"
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        main.enc = None
        with patch('main.tiktoken.get_encoding', return_value=mock_encoding):
            tokens = main.tokenizer(text)
        self.assertTrue(all(isinstance(token, int) for token in tokens))
        self.assertTrue(len(tokens) >= 2)

if __name__ == '__main__':
    unittest.main()
