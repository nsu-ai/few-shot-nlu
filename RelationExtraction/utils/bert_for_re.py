import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pickle
import os
from torch.nn import functional as F


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BertREConfig(BertConfig):
    def __init__(
            self,
            dropout_rate=0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dropout_rate = dropout_rate


def entity_average(hidden_output, e_mask):
    """
    Average the entity hidden state vectors (H_i ~ H_j)
    :param hidden_output: [batch_size, j-i+1, dim]
    :param e_mask: [batch_size, max_seq_len]
            e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
    :return: [batch_size, dim]
    """
    e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
    length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

    sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
    avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    return avg_vector


class BertRandomForestRE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.num_labels = config.num_labels

        self.rf = RandomForestClassifier(max_depth=15)

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        pickle.dump(self.rf, open(os.path.join(save_directory, "random_forest.sav"), 'wb'))

    def load_weights(self, model_path):
        self.rf = pickle.load(open(os.path.join(model_path, "random_forest.sav"), 'rb'))

    def convert_dataset(self, dataloader):
        X = []
        y = []
        for batch in tqdm(dataloader):
            outputs = self.bert(batch[0], attention_mask=batch[1],
                                token_type_ids=batch[2])
            sequence_output = outputs[0]

            # Average
            e1_h = entity_average(sequence_output, batch[4])
            e2_h = entity_average(sequence_output, batch[5])

            concat_h = torch.cat([e1_h, e2_h], dim=-1)

            X.append(concat_h.detach().numpy())
            y.append(batch[3].detach().numpy())

        return np.concatenate(X), np.concatenate(y)

    def fit(self, dataloader):
        self.rf = RandomForestClassifier(max_depth=5)
        X, y = self.convert_dataset(dataloader)
        self.rf.fit(X, y)

    def predict(self, X):
        return self.rf.predict(X)


class BertRE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.dropout_rate)
        self.e1_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.dropout_rate)
        self.e2_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size * 3, config.num_labels, config.dropout_rate, use_activation=False)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = entity_average(sequence_output, e1_mask)
        e2_h = entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertRELogisticRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.num_labels = config.num_labels

        self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]

        # Average
        e1_h = entity_average(sequence_output, e1_mask)
        e2_h = entity_average(sequence_output, e2_mask)

        concat_h = torch.cat([e1_h, e2_h], dim=-1)
        logits = F.sigmoid(self.linear(concat_h))

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
