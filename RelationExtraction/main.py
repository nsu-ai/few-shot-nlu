import os
from RelationExtraction.relation_extractor import RelationExtractor


if __name__ == "__main__":
    re = RelationExtractor(
        os.path.join("re_sci_data", "labels.txt"),
        model_type="BertRE"
        # use_cuda=True
    )

    """ Train example """
    # re.train(
    #     os.path.join("bert_base"),
    #     os.path.join("re_output"),
    #     os.path.join("re_sci_data"),
    #     save_steps=3,
    #     # per_gpu_train_batch_size=1
    # )

    """ Eval example """
    # re.eval(
    #     os.path.join("bert_base"),
    #     os.path.join("re_sci_data")
    # )

    """ Load example """
    re.load_model(
        os.path.join("re_output/checkpoint-3")
    )

    """ Predict example """
    sentences = [
        "Novosibirsk state university is located in the world-famous scientific center – Akademgorodok.",
        'NSU’s new building will remind you of R2D2, the astromech droid and hero of the Star Wars saga'
    ]
    entities = [
        {
            'LOCATION': [(80, 93)],
            'ORG': [(0, 28)]
        },
        {
            'ORG': [(0, 3)],
            'PERSON': [(38, 42)]
        }
    ]
    re.load_model(
        os.path.join("re_output", "checkpoint-3")
    )
    res = re.predict(
        sentences,
        entities
    )
    print(res)
