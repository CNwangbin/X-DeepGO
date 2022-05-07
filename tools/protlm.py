import sys

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from transformers import (BertConfig, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from deepfold.data.protein_dataset import ProtBertDataset
from deepfold.models.transformers.multilabel_transformer import \
    BertForMultiLabelSequenceClassification

sys.path.append('../')


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return {'auc': auc, 'ap': ap}


if __name__ == '__main__':
    model_name = 'Rostlab/prot_bert_bfd'
    data_root = '/home/niejianzheng/xbiome/datasets/protein'
    train_dataset = ProtBertDataset(
        data_path=data_root,
        split='train',
        tokenizer_name=model_name,
        max_length=512)  # max_length is only capped to speed-up example.
    val_dataset = ProtBertDataset(data_path=data_root,
                                  split='valid',
                                  tokenizer_name=model_name,
                                  max_length=512)
    test_dataset = ProtBertDataset(data_path=data_root,
                                   split='test',
                                   tokenizer_name=model_name,
                                   max_length=512)
    num_classes = train_dataset.num_classes
    model_config = BertConfig.from_pretrained(model_name,
                                              num_labels=num_classes)
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        model_name, config=model_config)

    # setting custom optimization parameters. You may implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    training_args = TrainingArguments(
        output_dir='./work_dir',  # output directory
        num_train_epochs=30,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,  # How often to print logs
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        evaluation_strategy='epoch',  # evalute after eachh epoch
        gradient_accumulation_steps=2,
        # total number of steps before back propagation
        fp16=True,  # Use mixed precision
        fp16_opt_level='02',  # mixed precision mode
        # report_to='wandb',  # enable logging to W&B
        run_name='ProBert-BFD-MS',  # experiment name
        seed=3  # Seed for experiment reproducibility 3x3
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        # compute_metrics=compute_metrics,  # evaluation metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model('models/')
    # Load trained model
    model_path = 'work_dir/checkpoint-5000'
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        model_path, num_labels=num_classes)
    # Define test trainer
    test_trainer = Trainer(model)
    predictions, label_ids, metrics = trainer.predict(val_dataset)
