import math
import torch

import transformers.activations
import transformers.modeling_utils
from transformers import GPT2LMHeadModel
from torch.nn import CrossEntropyLoss


def _gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def load_quantized_model(cls, path):
    # This hack is necessary in transformers 2.8.0 -- should revisit with a new release
    transformers.activations.ACT2FN["gelu_new"] = _gelu_new

    m = load_model(cls, path)
    qm = torch.quantization.quantize_dynamic(
        m, {torch.nn.Linear, torch.nn.Embedding, transformers.modeling_utils.Conv1D}, dtype=torch.qint8
    )
    return qm


def load_model(cls, path):
    return cls.from_pretrained(path)


class GPT2LMHeadWithWeightedLossModel(GPT2LMHeadModel):
    # See https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_weights=None,
    ):
        if loss_weights is not None and labels is None:
            raise RuntimeError("Labels must be specified to use loss weights")

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_loss_weights = loss_weights[..., 1:].contiguous()

            # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss_fct_flat = CrossEntropyLoss(reduction="none")
            loss_flat = loss_fct_flat(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weighted_loss_flat = shift_loss_weights.view(-1) * loss_flat
            weighted_loss = torch.mean(weighted_loss_flat)

            outputs = (weighted_loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
