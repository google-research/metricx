# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression checks for the two alternative encoder input paths."""

from unittest import mock

from metricx23 import models as models23
from metricx24 import models as models24
import pytest
import torch
from transformers import MT5Config


@pytest.fixture(params=[models23, models24], ids=["metricx23", "metricx24"])
def model(request):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        config = MT5Config(vocab_size=250100,
                           d_model=8,
                           d_kv=4,
                           d_ff=16,
                           num_layers=1,
                           num_decoder_layers=1,
                           num_heads=2,
                           dropout_rate=0.,
                           tie_word_embeddings=False)
        network = request.param.MT5ForRegression(config).eval()
    return network


def inputs(batch_size=2):
    ids = torch.arange(2, 2 + batch_size * 4).reshape(batch_size, 4)
    mask = torch.ones_like(ids)
    mask[:, -1] = 0
    return ids, mask


def calibrate_head(model, ids, mask):
    """Keep the tested prediction inside the existing clipping interval."""
    with torch.no_grad():
        hidden = model.encoder(input_ids=ids, attention_mask=mask)[0]
        decoded = model.decoder(input_ids=torch.zeros((ids.shape[0], 1),
                                                      dtype=torch.long),
                                encoder_hidden_states=hidden,
                                encoder_attention_mask=mask)[0][0, 0]
        model.lm_head.weight[250089].copy_(5. * decoded /
                                           decoded.square().sum())


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("return_dict", [True, False])
@pytest.mark.parametrize("mode",
                         ["embeddings", "cached_output", "cached_tuple"])
def test_alternative_input_matches_token_ids(model, batch_size, return_dict,
                                             mode):
    ids, mask = inputs(batch_size)
    calibrate_head(model, ids, mask)
    labels = torch.arange(batch_size, dtype=torch.float32) + 1.
    with torch.no_grad():
        expected = model(input_ids=ids,
                         attention_mask=mask,
                         labels=labels,
                         return_dict=return_dict)
        assert 0. < expected.predictions[0] < 25.
        if mode == "embeddings":
            kwargs = {"inputs_embeds": model.shared(ids)}
        else:
            encoded = model.encoder(input_ids=ids,
                                    attention_mask=mask,
                                    output_hidden_states=True,
                                    output_attentions=True)
            kwargs = {
                "encoder_outputs":
                    (encoded.to_tuple() if mode == "cached_tuple" else encoded)
            }
        with mock.patch.object(model.encoder,
                               "forward",
                               wraps=model.encoder.forward) as encoder:
            actual = model(attention_mask=mask,
                           labels=labels,
                           return_dict=return_dict,
                           **kwargs)
            assert encoder.call_count == (1 if mode == "embeddings" else 0)
    assert actual.predictions.shape == (batch_size,)
    torch.testing.assert_close(actual.predictions,
                               expected.predictions,
                               rtol=0.,
                               atol=0.)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0., atol=0.)


@pytest.mark.parametrize("mode", ["embeddings", "cached_output"])
def test_input_gradients_are_preserved(model, mode):
    ids, mask = inputs()
    calibrate_head(model, ids, mask)
    labels = torch.ones(ids.shape[0])
    reference_inputs = model.shared(ids).detach().requires_grad_()
    reference_encoder = model.encoder(inputs_embeds=reference_inputs,
                                      attention_mask=mask)
    expected = model(input_ids=ids,
                     encoder_outputs=reference_encoder,
                     attention_mask=mask,
                     labels=labels)
    expected_gradient, = torch.autograd.grad(expected.loss, reference_inputs)
    assert torch.isfinite(expected_gradient).all()
    assert torch.count_nonzero(expected_gradient) > 0

    actual_inputs = reference_inputs.detach().clone().requires_grad_()
    if mode == "embeddings":
        kwargs = {"inputs_embeds": actual_inputs}
    else:
        kwargs = {
            "encoder_outputs":
                model.encoder(inputs_embeds=actual_inputs, attention_mask=mask)
        }
    actual = model(attention_mask=mask, labels=labels, **kwargs)
    actual_gradient, = torch.autograd.grad(actual.loss, actual_inputs)
    torch.testing.assert_close(actual.predictions,
                               expected.predictions,
                               rtol=0.,
                               atol=0.)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0., atol=0.)
    torch.testing.assert_close(actual_gradient,
                               expected_gradient,
                               rtol=0.,
                               atol=0.)


def test_conflicting_encoder_inputs_still_raise(model):
    ids, _ = inputs()
    with pytest.raises(ValueError, match="both"):
        model(input_ids=ids, inputs_embeds=model.shared(ids))


def test_missing_encoder_inputs_still_raise(model):
    with pytest.raises(ValueError, match="specify"):
        model()


def test_local_checkpoint_retains_alternative_input_support(model, tmp_path):
    ids, mask = inputs()
    calibrate_head(model, ids, mask)
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask).predictions
        model.save_pretrained(tmp_path, safe_serialization=True)
        restored = type(model).from_pretrained(tmp_path,
                                               local_files_only=True).eval()
        actual = restored(inputs_embeds=restored.shared(ids),
                          attention_mask=mask).predictions
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
