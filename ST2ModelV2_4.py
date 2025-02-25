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
    @classmethod
    def from_pretrained(cls, model_name, args=None, **kwargs):
        """
        Custom from_pretrained method to load model from Hugging Face.
        """
        # Load configuration
        config = AutoConfig.from_pretrained("roberta-large")
        
        # Instantiate the model
        model = cls(args, config)

        # Load pre-trained weights from Hugging Face
        model.model = AutoModel.from_pretrained(model_name, config=config)

        return model
        
    def __init__(self, args, config):
        super(ST2ModelV2, self).__init__()
        self.args = args
        self.config = config

        # Ensure hidden_size exists
        self.config = AutoConfig.from_pretrained("roberta-large")

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

    def position_selector(
        self,
        start_cause_logits, 
        start_effect_logits, 
        end_cause_logits, 
        end_effect_logits,
        attention_mask,
        word_ids,
     ):
        # basic post processing (removing logits from [CLS], [SEP], [PAD])
        start_cause_logits -= (1 - attention_mask) * 1e4
        end_cause_logits -= (1 - attention_mask) * 1e4
        start_effect_logits -= (1 - attention_mask) * 1e4
        end_effect_logits -= (1 - attention_mask) * 1e4

        start_cause_logits[0] = -1e4
        end_cause_logits[0] = -1e4
        start_effect_logits[0] = -1e4
        end_effect_logits[0] = -1e4

        start_cause_logits[len(word_ids) - 1] = -1e4
        end_cause_logits[len(word_ids) - 1] = -1e4
        start_effect_logits[len(word_ids) - 1] = -1e4
        end_effect_logits[len(word_ids) - 1] = -1e4

        start_cause_logits = torch.log(torch.softmax(start_cause_logits, dim=-1))
        end_cause_logits = torch.log(torch.softmax(end_cause_logits, dim=-1))
        start_effect_logits = torch.log(torch.softmax(start_effect_logits, dim=-1))
        end_effect_logits = torch.log(torch.softmax(end_effect_logits, dim=-1))

        max_arg0_before_arg1 = None
        for i in range(len(end_cause_logits)):
            if attention_mask[i] == 0:
                break
            for j in range(i + 1, len(start_effect_logits)):
                if attention_mask[j] == 0:
                    break

                if max_arg0_before_arg1 is None:
                    max_arg0_before_arg1 = ((i, j), end_cause_logits[i] + start_effect_logits[j])
                else:
                    if end_cause_logits[i] + start_effect_logits[j] > max_arg0_before_arg1[1]:
                        max_arg0_before_arg1 = ((i, j), end_cause_logits[i] + start_effect_logits[j])
        
        max_arg0_after_arg1 = None
        for i in range(len(end_effect_logits)):
            if attention_mask[i] == 0:
                break
            for j in range(i + 1, len(start_cause_logits)):
                if attention_mask[j] == 0:
                    break
                if max_arg0_after_arg1 is None:
                    max_arg0_after_arg1 = ((i, j), start_cause_logits[j] + end_effect_logits[i])
                else:
                    if start_cause_logits[j] + end_effect_logits[i] > max_arg0_after_arg1[1]:
                        max_arg0_after_arg1 = ((i, j), start_cause_logits[j] + end_effect_logits[i])

        if max_arg0_before_arg1[1].item() > max_arg0_after_arg1[1].item():
            end_cause, start_effect = max_arg0_before_arg1[0]
            start_cause_logits[end_cause + 1:] = -1e4
            start_cause = start_cause_logits.argmax().item()

            end_effect_logits[:start_effect] = -1e4
            end_effect = end_effect_logits.argmax().item()
        else:
            end_effect, start_cause = max_arg0_after_arg1[0]
            end_cause_logits[:start_cause] = -1e4
            end_cause = end_cause_logits.argmax().item()

            start_effect_logits[end_effect + 1:] = -1e4
            start_effect = start_effect_logits.argmax().item()
        
        return start_cause, end_cause, start_effect, end_effect


    def beam_search_position_selector(
        self,
        start_cause_logits, 
        start_effect_logits, 
        end_cause_logits, 
        end_effect_logits,
        topk=5
     ):
        
        start_cause_logits = torch.log(torch.softmax(start_cause_logits, dim=-1))
        end_cause_logits = torch.log(torch.softmax(end_cause_logits, dim=-1))
        start_effect_logits = torch.log(torch.softmax(start_effect_logits, dim=-1))
        end_effect_logits = torch.log(torch.softmax(end_effect_logits, dim=-1))

        scores = dict()
        for i in range(len(end_cause_logits)):
            
            for j in range(i + 1, len(start_effect_logits)):
                scores[str((i, j, "before"))] = end_cause_logits[i].item() + start_effect_logits[j].item()
        
        for i in range(len(end_effect_logits)):
            for j in range(i + 1, len(start_cause_logits)):
                scores[str((i, j, "after"))] = start_cause_logits[j].item() + end_effect_logits[i].item()
        
        
        topk_scores = dict()
        for i, (index, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]):
            if eval(index)[2] == 'before':
                end_cause = eval(index)[0]
                start_effect = eval(index)[1]

                this_start_cause_logits = start_cause_logits.clone()
                this_start_cause_logits[end_cause + 1:] = -1e9
                start_cause_values, start_cause_indices = this_start_cause_logits.topk(topk)

                this_end_effect_logits = end_effect_logits.clone()
                this_end_effect_logits[:start_effect] = -1e9
                end_effect_values, end_effect_indices = this_end_effect_logits.topk(topk)

                for m in range(len(start_cause_values)):
                    for n in range(len(end_effect_values)):
                        topk_scores[str((start_cause_indices[m].item(), end_cause, start_effect, end_effect_indices[n].item()))] = score + start_cause_values[m].item() + end_effect_values[n].item()

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
