"""Run with python -m unittest discover -s tests -p test_pi05_rgb.py."""
import tempfile
import unittest
from unittest.mock import patch

import torch
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import TransitionKey as K
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from evo_rlt.adapters.lerobot.policies.processor_pi05_rgb import Pi05RGBTokenizerStep


class FakeTokenizer:
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]


class RGBTests(unittest.TestCase):
    def test_state_invariance_and_checkpoint_roundtrip(self):
        with patch('evo_rlt.adapters.lerobot.policies.processor_pi05_rgb.AutoTokenizer.from_pretrained',
                   return_value=FakeTokenizer()):
            step = Pi05RGBTokenizerStep()
            outputs = []
            for state in (None, torch.zeros(1, 15), torch.full((1, 15), 999.0)):
                obs = {} if state is None else {'observation.state': state}
                transition = {K.OBSERVATION: obs, K.COMPLEMENTARY_DATA: {'task': ['pick_up cable']}}
                outputs.append(step(transition)[K.OBSERVATION])
                self.assertEqual(transition[K.COMPLEMENTARY_DATA]['task'], ['pick_up cable'])
            for obs in outputs:
                self.assertNotIn('observation.state', obs)
                for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
                    self.assertTrue(torch.equal(obs[key], outputs[0][key]))
            expected = [2] + [ord(c) for c in 'pick up cable'] + [10]
            self.assertEqual(outputs[0][OBS_LANGUAGE_TOKENS][0, :len(expected)].tolist(), expected)
            with tempfile.TemporaryDirectory() as tmp:
                pipeline = PolicyProcessorPipeline(steps=[step], name='policy_preprocessor')
                pipeline.save_pretrained(tmp)
                restored = PolicyProcessorPipeline.from_pretrained(tmp, config_filename='policy_preprocessor.json')
                result = restored.steps[0]({K.OBSERVATION: {}, K.COMPLEMENTARY_DATA: {'task': ['pick_up cable']}})
                self.assertFalse(restored.steps[0].discrete_state_input)
                self.assertTrue(torch.equal(result[K.OBSERVATION][OBS_LANGUAGE_TOKENS], outputs[0][OBS_LANGUAGE_TOKENS]))


if __name__ == '__main__':
    unittest.main()
