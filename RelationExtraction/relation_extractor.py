import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AdamW,
    BertTokenizer,
)
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
from .utils.utils_re import convert_examples_to_features, get_labels, read_examples_from_file, ADDITIONAL_SPECIAL_TOKENS
from .utils.bert_for_re import BertRE, BertREConfig, BertRELogisticRegression, BertRandomForestRE
from tensorboardX import SummaryWriter


class RelationExtractor:

    def __init__(self,
                 path_to_labels,  # Path to a file containing all labels. If not specified, CoNLL-2003 labels are used
                 model_type="BertRE",  # Can be "BertRE" or "BertRELogisticRegression"
                 max_seq_len=128,  # The maximum total input sequence length after tokenization. Sequences longer
                 # than this will be truncated, sequences shorter will be padded
                 use_cuda=False,  # Using CUDA when available
                 seed=42,  # random seed for initialization
                 # Tokenizer args
                 tokenizer_name=None,  # Pre-trained tokenizer name or path if not the same as model_name
                 do_lower_case=True,  # Set this flag if you are using an uncased model
                 keep_accents=True,  # Set this flag if model is trained with accents
                 strip_accents=True,  # Set this flag if model is trained without accents
                 use_fast=True  # Set this flag to use fast tokenization
                 ):

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self._n_gpu = 0 if not use_cuda else torch.cuda.device_count()
        self._model_type = model_type
        self._model = None
        self._tokenizer = None

        self._max_seq_len = max_seq_len
        self._tokenizer_args = {
            "tokenizer_name": tokenizer_name,
            "do_lower_case": do_lower_case,
            "keep_accents": keep_accents,
            "strip_accents": strip_accents,
            "use_fast": use_fast
        }

        self._logger = logging.getLogger()
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._labels = get_labels(path_to_labels)

    def train(self,
              model_path,  # Path to pre-trained model
              output_dir,  # The output directory where the model predictions and checkpoints will be written
              data_dir,  # The input data dir. Should contain the training files
              cache_dir=None,  # Where do you want to store the pre-trained models downloaded from s3
              overwrite_output_dir=True,  # Overwrite the content of the output directory
              per_gpu_train_batch_size=8,  # Batch size per GPU/CPU for training
              learning_rate=0.00005,  # The initial learning rate for Adam
              weight_decay=0.0,  # Weight decay if we apply some
              adam_epsilon=0.00000001,  # Epsilon for Adam optimizer
              max_grad_norm=1.0,  # Max gradient norm
              num_train_epochs=3.0,  # Total number of training epochs to perform
              max_steps=-1,  # If > 0: set total number of training steps to perform. Override num_train_epochs
              warmup_steps=0,  # Linear warmup over warmup_steps
              gradient_accumulation_steps=1,
              # Number of updates steps to accumulate before performing a backward/update pass
              logging_steps=500,  # Log every X updates steps
              save_steps=500,  # Save checkpoint every X updates steps
              dropout_rate=0.1,  # Dropout for model
              ):
        model_config = BertREConfig.from_pretrained(
            model_path,
            num_labels=len(self._labels),
            dropout_rate=dropout_rate
        )

        self._logger.info("Tokenizer arguments: %s", self._tokenizer_args)
        tokenizer = BertTokenizer.from_pretrained(
            self._tokenizer_args["tokenizer_name"] if self._tokenizer_args["tokenizer_name"] else model_path,
            cache_dir=cache_dir,
            **self._tokenizer_args,
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

        model = None
        if self._model_type == "BertRE":
            model = BertRE.from_pretrained(
                model_path,
                config=model_config,
                cache_dir=cache_dir,
            )
            model.to(self._device)
        elif self._model_type == "BertRELogisticRegression":
            model = BertRELogisticRegression.from_pretrained(
                model_path,
                config=model_config,
                cache_dir=cache_dir,
            )
            model.to(self._device)
        elif self._model_type == "BertRandomForestRE":
            model = BertRandomForestRE.from_pretrained(model_path, config=model_config)

        if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    output_dir
                )
            )

        train_dataset = self.__load_and_cache_examples(model_path, tokenizer, data_dir, mode="train")
        tb_writer = SummaryWriter()
        train_batch_size = per_gpu_train_batch_size * max(1, self._n_gpu)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if self._model_type == "BertRandomForestRE":
            model.fit(train_dataloader)
            self._tokenizer = tokenizer
            self._model = model
            self._model.save_pretrained(output_dir)
            self._tokenizer.save_pretrained(output_dir)
            return

        """
        Next - training neural network models 
        """

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                          eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps=warmup_steps, t_total=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(model_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(model_path, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        if self._n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Train!
        self._logger.info("***** Running training *****")
        self._logger.info("  Num examples = %d", len(train_dataset))
        self._logger.info("  Num Epochs = %d", num_train_epochs)
        self._logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)
        self._logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        self._logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(model_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(model_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // gradient_accumulation_steps)

            self._logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self._logger.info("  Continuing training from epoch %d", epochs_trained)
            self._logger.info("  Continuing training from global step %d", global_step)
            self._logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch"
                                )
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self._device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self._n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr(), global_step)
                        tb_writer.add_scalar("loss",
                                             (tr_loss - logging_loss) / logging_steps, global_step)
                        logging_loss = tr_loss

                    if save_steps > 0 and global_step % save_steps == 0:
                        # Save model checkpoint
                        out_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(out_dir)
                        tokenizer.save_pretrained(out_dir)

                        self._logger.info("Saving model checkpoint to %s", out_dir)
                        torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(out_dir, "scheduler.pt"))
                        self._logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        tb_writer.close()

        self._logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        self._tokenizer = tokenizer
        self._model = model

    def eval(self,
             model_path,
             data_dir,
             per_gpu_eval_batch_size=8  # Batch size per GPU/CPU for evaluation
             ):
        # if self._tokenizer is None:
        #     self._tokenizer = BertTokenizer.from_pretrained(output_dir)
        # checkpoint = output_dir
        # self._logger.info("Evaluate the following checkpoint: %s", checkpoint)
        # global_step = ""
        #
        # if self._model == None:
        #     if self._model_type == "BertRE":
        #         model = BertRE.from_pretrained(checkpoint)
        #     elif self._model_type == "BertRELogisticRegression":
        #         model = BertRELogisticRegression.from_pretrained(checkpoint)
        #     model.to(self._device)

        eval_dataset = self.__load_and_cache_examples(model_path, self._tokenizer, data_dir, mode="dev")
        global_step = 0

        eval_batch_size = per_gpu_eval_batch_size * max(1, self._n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if self._model_type == "BertRandomForestRE":
            X, y = self._model.convert_dataset(eval_dataloader)
            preds = self._model.predict(X)
            res = precision_recall_fscore_support(y, preds, average='macro')
            results = {
                "precision": res[0],
                "recall": res[1],
                "f1": res[2],
            }

            self._logger.info("***** Eval results %s *****", global_step)
            for key in sorted(results.keys()):
                self._logger.info("  %s = %s", key, str(results[key]))
            return

        # multi-gpu evaluate
        if self._n_gpu > 1:
            self._model = torch.nn.DataParallel(self._model)

        # Eval!
        self._logger.info("***** Running evaluation %s *****", global_step)
        self._logger.info("  Num examples = %d", len(eval_dataset))
        self._logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self._model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self._device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                outputs = self._model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if self._n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        res = precision_recall_fscore_support(out_label_ids, preds, average='macro')
        results = {
            "loss": eval_loss,
            "precision": res[0],
            "recall": res[1],
            "f1": res[2],
        }

        self._logger.info("***** Eval results %s *****", global_step)
        for key in sorted(results.keys()):
            self._logger.info("  %s = %s", key, str(results[key]))

    """
    Input data format
    
    List of strings:
    [
    'Novosibirsk state university is located in the world-famous scientific center – Akademgorodok.',
    '"It’s like in the woods!" – that's what people say when they come to Akademgorodok for the first time.',
    'NSU’s new building will remind you of R2D2, the astromech droid and hero of the Star Wars saga'
    ]
    List of entities:
    [
        {
            'LOCATION': [(80, 93)],
            'ORG': [(0, 28)]
        },
        {
            'LOCATION': [(69, 82)]
        },
        {
            'ORG': [(0, 3)],
            'PERSON': [(38, 42)]
        }
    ]
    """

    def predict(self,
                sentences,
                entities,
                ):
        special_tokens_count = 1
        cls_token = '[CLS]'
        cls_token_segment_id = 0
        sep_token = '[SEP]'
        pad_token = 0
        pad_token_segment_id = 0
        sequence_a_segment_id = 0
        add_sep_token = False
        mask_padding_with_zero = True

        inputs = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'e1_mask': [],
            'e2_mask': []
        }

        for i in range(len(sentences)):
            first = (list(entities[i].keys())[0], entities[i][list(entities[i].keys())[0]][0])
            second = (list(entities[i].keys())[1], entities[i][list(entities[i].keys())[1]][0])

            # This is what real laziness looks like
            if first[1][1] < second[1][0]:
                text = sentences[i][:first[1][0]] + "<e1>" + \
                       sentences[i][first[1][0]:first[1][1]] + "</e1>" + \
                       sentences[i][first[1][1]:second[1][0]] + "<e2>" + \
                       sentences[i][second[1][0]:second[1][1]] + "</e2>" + \
                       sentences[i][second[1][1]:]
            elif first[1][0] > second[1][1]:
                text = sentences[i][:second[1][0]] + "<e2>" + \
                       sentences[i][second[1][0]:second[1][1]] + "</e2>" + \
                       sentences[i][second[1][1]:first[1][0]] + "<e1>" + \
                       sentences[i][first[1][0]:first[1][1]] + "</e1>" + \
                       sentences[i][first[1][1]:]
            elif second[1][0] > first[1][0] and second[1][1] < first[1][1]:
                text = sentences[i][:first[1][0]] + "<e1>" + \
                       sentences[i][first[1][0]:second[1][0]] + "<e2>" + \
                       sentences[i][second[1][0]:second[1][1]] + "</e2>" + \
                       sentences[i][second[1][1]:first[1][1]] + "</e1>" + \
                       sentences[i][first[1][1]:]
            elif second[1][0] < first[1][0] and second[1][1] > first[1][1]:
                text = sentences[i][:second[1][0]] + "<e2>" + \
                       sentences[i][second[1][0]:first[1][0]] + "<e1>" + \
                       sentences[i][first[1][0]:first[1][1]] + "</e1>" + \
                       sentences[i][first[1][1]:second[1][1]] + "</e2>" + \
                       sentences[i][second[1][1]:]
            elif first[1][0] < second[1][0] < first[1][1] < second[1][1]:
                text = sentences[i][:first[1][0]] + "<e1>" + \
                       sentences[i][first[1][0]:second[1][0]] + "<e2>" + \
                       sentences[i][second[1][0]:first[1][1]] + "</e1>" + \
                       sentences[i][first[1][1]:second[1][1]] + "</e2>" + \
                       sentences[i][second[1][1]:]
            elif second[1][0] < first[1][0] < second[1][1] < first[1][1]:
                text = sentences[i][:second[1][0]] + "<e2>" + \
                       sentences[i][second[1][0]:first[1][0]] + "<e1>" + \
                       sentences[i][first[1][0]:second[1][1]] + "</e2>" + \
                       sentences[i][second[1][1]:first[1][1]] + "</e1>" + \
                       sentences[i][first[1][1]:]
            else:
                raise Exception("Fufk, something wrong")

            tokens_a = self._tokenizer.tokenize(text)

            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            if len(tokens_a) > self._max_seq_len - special_tokens_count:
                tokens_a = tokens_a[:(self._max_seq_len - special_tokens_count)]

            tokens = tokens_a
            if add_sep_token:
                tokens += [sep_token]

            token_type_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for j in range(e11_p, e12_p + 1):
                e1_mask[j] = 1
            for j in range(e21_p, e22_p + 1):
                e2_mask[j] = 1

            assert len(input_ids) == self._max_seq_len, "Error with input length {} vs {}".format(
                len(input_ids), self._max_seq_len)
            assert len(attention_mask) == self._max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), self._max_seq_len)
            assert len(token_type_ids) == self._max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), self._max_seq_len)

            inputs['input_ids'].append(input_ids)
            inputs['attention_mask'].append(attention_mask)
            inputs['token_type_ids'].append(token_type_ids)
            inputs['e1_mask'].append(e1_mask)
            inputs['e2_mask'].append(e2_mask)

        data = {'input_ids': torch.tensor([x for x in inputs["input_ids"]], dtype=torch.long),
                'attention_mask': torch.tensor([x for x in inputs["attention_mask"]], dtype=torch.long),
                'token_type_ids': torch.tensor([x for x in inputs["token_type_ids"]], dtype=torch.long),
                'labels': None,
                'e1_mask': torch.tensor([x for x in inputs["e1_mask"]], dtype=torch.long),
                'e2_mask': torch.tensor([x for x in inputs["e2_mask"]], dtype=torch.long)}

        label_probs = self._model(**data)[0]
        label_ids = torch.argmax(label_probs, dim=1)
        outputs = [self._labels[i] for i in label_ids]

        return outputs

    def load_model(self,
                   model_path
                   ):
        self._tokenizer = BertTokenizer.from_pretrained(model_path)
        self._model = None
        if self._model_type == "BertRE":
            self._model = BertRE.from_pretrained(model_path)
            self._model.to(self._device)
        elif self._model_type == "BertRELogisticRegression":
            self._model = BertRELogisticRegression.from_pretrained(model_path)
        elif self._model_type == "BertRandomForestRE":
            self._model = BertRandomForestRE.from_pretrained(model_path)
            self._model.load_weights(model_path)

    def __load_and_cache_examples(self,
                                  model_path,
                                  tokenizer,
                                  data_dir,
                                  mode):
        cached_features_file = os.path.join(
            data_dir,
            'cached_{}_{}_{}'.format(
                mode,
                list(filter(None, model_path.split("/"))).pop(),
                str(self._max_seq_len)
            )
        )
        if os.path.exists(cached_features_file):
            self._logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            self._logger.info("Creating features from dataset file at %s", data_dir)
            examples = read_examples_from_file(data_dir, mode)
            features = convert_examples_to_features(examples, self._max_seq_len, tokenizer,
                                                    self._labels)
            self._logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
        return dataset
