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
        config = AutoConfig.from_pretrained(model_name)
        
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
        self.config = AutoConfig.from_pretrained(args.model_name)

        # Define classifier layers
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

        if signal_bias_mask is not None and self.args.add_signal_bias:
            sequence_output[signal_bias_mask == 1] += self.signal_phrases_layer

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # Split logits for different tasks
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

        # Compute loss if labels are provided
        loss_fct = nn.CrossEntropyLoss()
        total_loss = None

        if start_positions is not None and end_positions is not None:
            start_arg0_loss = loss_fct(start_arg0_logits, start_positions[:, 0])
            end_arg0_loss = loss_fct(end_arg0_logits, end_positions[:, 0])
            arg0_loss = (start_arg0_loss + end_arg0_loss) / 2

            start_arg1_loss = loss_fct(start_arg1_logits, start_positions[:, 1])
            end_arg1_loss = loss_fct(end_arg1_logits, end_positions[:, 1])
            arg1_loss = (start_arg1_loss + end_arg1_loss) / 2

            start_sig_loss = loss_fct(start_sig_logits, start_positions[:, 2])
            end_sig_loss = loss_fct(end_sig_logits, end_positions[:, 2])
            sig_loss = (start_sig_loss + end_sig_loss) / 2

            if self.args.signal_classification and not self.args.pretrained_signal_detector:
                signal_classification_labels = (end_positions[:, 2] != -100).long()
                signal_classification_loss = loss_fct(signal_classification_logits, signal_classification_labels)
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
            'loss': total_loss,
        }

