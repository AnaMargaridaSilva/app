import torch
from torch import nn
from typing import Optional
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification
    )
from statistics import mode
from safetensors.torch import load_file



class ST2ModelV2(nn.Module):
    def __init__(self, args, config):
        super(ST2ModelV2, self).__init__()
        self.args = args
        self.config = config

        # Load the base model (e.g., Roberta)
        self.model = AutoModel.from_pretrained("roberta-large", config=config)

        # Define classifier layers
        classifier_dropout = self.args.dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 6)

        if self.args.signal_classification and not self.args.pretrained_signal_detector:
            self.signal_classifier = nn.Linear(self.config.hidden_size, 2)

  

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        signal_bias_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Ensure that self.model is not None before calling forward
        if self.model is None:
            raise ValueError("The model weights have not been loaded. Use from_pretrained() to load them.")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # Split logits 
        start_arg0_logits, end_arg0_logits, start_arg1_logits, end_arg1_logits, start_sig_logits, end_sig_logits = logits.split(1, dim=-1)
        start_arg0_logits = start_arg0_logits.squeeze(-1)
        end_arg0_logits = end_arg0_logits.squeeze(-1)
        start_arg1_logits = start_arg1_logits.squeeze(-1)
        end_arg1_logits = end_arg1_logits.squeeze(-1)
        start_sig_logits = start_sig_logits.squeeze(-1)
        end_sig_logits = end_sig_logits.squeeze(-1)

        signal_classification_logits = None
        if self.args.signal_classification and not self.args.pretrained_signal_detector:
            signal_classification_logits = self.signal_classifier(sequence_output[:, 0, :])

        return {
            'start_arg0_logits': start_arg0_logits,
            'end_arg0_logits': end_arg0_logits,
            'start_arg1_logits': start_arg1_logits,
            'end_arg1_logits': end_arg1_logits,
            'start_sig_logits': start_sig_logits,
            'end_sig_logits': end_sig_logits,
            'signal_classification_logits': signal_classification_logits
        }

    @classmethod
    def from_pretrained(cls, model_name, config=None, args=None, **kwargs):
        """
        Custom from_pretrained method to load the model from Hugging Face and initialize
        any additional components such as the classifier.
        """
        # Load the configuration
        config = AutoConfig.from_pretrained(model_name) if config is None else config
        
        # Instantiate the model
        model = cls(args, config)
        
        # Load the pre-trained weights into the model
        model.model = AutoModel.from_pretrained(model_name, config=config, **kwargs, use_safetensors=False)
        
        return model

    def beam_search_position_selector(
        self,
        start_cause_logits, 
        start_effect_logits, 
        end_cause_logits, 
        end_effect_logits,
        topk=5
    ):
        """
        Performs beam search to find the best positions for argument extraction.
        """
        start_cause_logits = torch.log(torch.softmax(start_cause_logits, dim=-1))
        end_cause_logits = torch.log(torch.softmax(end_cause_logits, dim=-1))
        start_effect_logits = torch.log(torch.softmax(start_effect_logits, dim=-1))
        end_effect_logits = torch.log(torch.softmax(end_effect_logits, dim=-1))

        scores = {}
        for i in range(len(end_cause_logits)):
            for j in range(i + 1, len(start_effect_logits)):
                scores[str((i, j, "before"))] = end_cause_logits[i].item() + start_effect_logits[j].item()
        
        for i in range(len(end_effect_logits)):
            for j in range(i + 1, len(start_cause_logits)):
                scores[str((i, j, "after"))] = start_cause_logits[j].item() + end_effect_logits[i].item()
        
        # Get top-k scores
        topk_scores = {}
        for i, (index, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]):
            index_tuple = eval(index)
            if index_tuple[2] == 'before':
                end_cause = index_tuple[0]
                start_effect = index_tuple[1]

                this_start_cause_logits = start_cause_logits.clone()
                this_start_cause_logits[end_cause + 1:] = -1e9
                start_cause_values, start_cause_indices = this_start_cause_logits.topk(topk)

                this_end_effect_logits = end_effect_logits.clone()
                this_end_effect_logits[:start_effect] = -1e9
                end_effect_values, end_effect_indices = this_end_effect_logits.topk(topk)

                for m in range(len(start_cause_values)):
                    for n in range(len(end_effect_values)):
                        topk_scores[str((start_cause_indices[m].item(), end_cause, start_effect, end_effect_indices[n].item()))] = (
                            score + start_cause_values[m].item() + end_effect_values[n].item()
                        )

            elif index_tuple[2] == 'after':
                start_cause = index_tuple[1]
                end_effect = index_tuple[0]

                this_end_cause_logits = end_cause_logits.clone()
                this_end_cause_logits[:start_cause] = -1e9
                end_cause_values, end_cause_indices = this_end_cause_logits.topk(topk)

                this_start_effect_logits = start_effect_logits.clone()
                this_start_effect_logits[end_effect + 1:] = -1e9
                start_effect_values, start_effect_indices = this_start_effect_logits.topk(topk)

                for m in range(len(end_cause_values)):
                    for n in range(len(start_effect_values)):
                        topk_scores[str((start_cause, end_cause_indices[m].item(), start_effect_indices[n].item(), end_effect))] = (
                            score + end_cause_values[m].item() + start_effect_values[n].item()
                        )

        first, second = sorted(topk_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        return eval(first[0]), eval(second[0]), first[1], second[1], topk_scores

            elif eval(index)[2] == 'after':
                start_cause = eval(index)[1]
                end_effect = eval(index)[0]

                this_end_cause_logits = end_cause_logits.clone()
                this_end_cause_logits[:start_cause] = -1e9
                end_cause_values, end_cause_indices = this_end_cause_logits.topk(topk)

                this_start_effect_logits = start_effect_logits.clone()
                this_start_effect_logits[end_effect + 1:] = -1e9
                start_effect_values, start_effect_indices = this_start_effect_logits.topk(topk)

                for m in range(len(end_cause_values)):
                    for n in range(len(start_effect_values)):
                        topk_scores[str((start_cause, end_cause_indices[m].item(), start_effect_indices[n].item(), end_effect))] = score + end_cause_values[m].item() + start_effect_values[n].item()

        first, second = sorted(topk_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        return eval(first[0]), eval(second[0]), first[1], second[1], topk_scores
