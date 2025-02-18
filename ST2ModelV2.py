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


class ST2ModelV2(nn.Module):
    def __init__(self, model_name):
        super(ST2ModelV2, self).__init__()
        

        
        self.model = AutoModel.from_pretrained(model_name)
        

        classifier_dropout = 0.3
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 6)

        if args.mlp:
            self.classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, 6),
                nn.Tanh(),
                nn.Linear(6, 6),
            )

        if args.add_signal_bias:
            self.signal_phrases_layer = nn.Parameter(
                torch.normal(
                    mean=self.model.embeddings.word_embeddings.weight.data.mean(), 
                    std=self.model.embeddings.word_embeddings.weight.data.std(),
                    size=(1, self.config.hidden_size),
                )
            )
        
        if self.args.signal_classification and not self.args.pretrained_signal_detector:
            self.signal_classifier = nn.Linear(self.config.hidden_size, 2)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        signal_bias_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,     # [batch_size, 3]
        end_positions: Optional[torch.Tensor] = None,       # [batch_size, 3]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if signal_bias_mask is not None and not self.args.signal_bias_on_top_of_lm:
            inputs_embeds = self.signal_phrases_bias(input_ids, signal_bias_mask)

            outputs = self.model(
                # input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            if self.args.model_name_or_path in ['facebook/bart-large', 'facebook/bart-base', 'facebook/bart-large-cnn']:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif self.args.model_name_or_path in ['microsoft/deberta-base']:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            else:               
                outputs = self.model(
                    input_ids,
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
        if signal_bias_mask is not None and self.args.signal_bias_on_top_of_lm:
            sequence_output[signal_bias_mask == 1] += self.signal_phrases_layer

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, max_seq_length, 6]
        start_arg0_logits, end_arg0_logits, start_arg1_logits, end_arg1_logits, start_sig_logits, end_sig_logits = logits.split(1, dim=-1)
        start_arg0_logits = start_arg0_logits.squeeze(-1).contiguous()
        end_arg0_logits = end_arg0_logits.squeeze(-1).contiguous()
        start_arg1_logits = start_arg1_logits.squeeze(-1).contiguous()
        end_arg1_logits = end_arg1_logits.squeeze(-1).contiguous()
        start_sig_logits = start_sig_logits.squeeze(-1).contiguous()
        end_sig_logits = end_sig_logits.squeeze(-1).contiguous()

        # start_arg0_logits -= (1 - attention_mask) * 1e4
        # end_arg0_logits -= (1 - attention_mask) * 1e4
        # start_arg1_logits -= (1 - attention_mask) * 1e4
        # end_arg1_logits -= (1 - attention_mask) * 1e4

        # start_arg0_logits[:, 0] = -1e4
        # end_arg0_logits[:, 0] = -1e4
        # start_arg1_logits[:, 0] = -1e4
        # end_arg1_logits[:, 0] = -1e4

        signal_classification_logits = None
        if self.args.signal_classification and not self.args.pretrained_signal_detector:
            signal_classification_logits = self.signal_classifier(sequence_output[:, 0, :])
        # start_logits = start_logits.squeeze(-1).contiguous()
        # end_logits = end_logits.squeeze(-1).contiguous()

        arg0_loss = None
        arg1_loss = None
        sig_loss = None
        total_loss = None
        signal_classification_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()

            start_arg0_loss = loss_fct(start_arg0_logits, start_positions[:, 0])
            end_arg0_loss = loss_fct(end_arg0_logits, end_positions[:, 0])
            arg0_loss = (start_arg0_loss + end_arg0_loss) / 2

            start_arg1_loss = loss_fct(start_arg1_logits, start_positions[:, 1])
            end_arg1_loss = loss_fct(end_arg1_logits, end_positions[:, 1])
            arg1_loss = (start_arg1_loss + end_arg1_loss) / 2

            # sig_loss = 0.
            start_sig_loss = loss_fct(start_sig_logits, start_positions[:, 2])
            end_sig_loss = loss_fct(end_sig_logits, end_positions[:, 2])
            sig_loss = (start_sig_loss + end_sig_loss) / 2

            if sig_loss.isnan():
                sig_loss = 0.

            if self.args.signal_classification and not self.args.pretrained_signal_detector:
                signal_classification_labels = end_positions[:, 2] != -100
                signal_classification_loss = loss_fct(signal_classification_logits, signal_classification_labels.long())
                total_loss = (arg0_loss + arg1_loss + sig_loss + signal_classification_loss) / 4
            else:
                total_loss = (arg0_loss + arg1_loss + sig_loss) / 3
                

        return {
            'start_arg0_logits': start_arg0_logits,
            'end_arg0_logits': end_arg0_logits,
            'start_arg1_logits': start_arg1_logits,
            'end_arg1_logits': end_arg1_logits,
            'start_sig_logits': start_sig_logits,
            'end_sig_logits': end_sig_logits,
            'signal_classification_logits': signal_classification_logits,
            'arg0_loss': arg0_loss,
            'arg1_loss': arg1_loss,
            'sig_loss': sig_loss,
            'signal_classification_loss': signal_classification_loss,
            'loss': total_loss,
        }
