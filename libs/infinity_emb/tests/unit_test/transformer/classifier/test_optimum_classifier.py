from transformers.pipelines import pipeline  # type: ignore
from infinity_emb.args import EngineArgs

from infinity_emb.inference.batch_handler import logits_to_probabilities
from infinity_emb.transformer.classifier.optimum import OptimumClassifier


def test_classifier(model_name: str = "SamLowe/roberta-base-go_emotions-onnx"):
    model = OptimumClassifier(
        engine_args=EngineArgs(
            model_name_or_path=model_name,
        )  # type: ignore
    )

    pipe = pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",  # hoping that this is the same model as model_name
        top_k=None,
    )

    sentences = ["This is awesome.", "I am depressed."]

    encode_pre = model.encode_pre(sentences)
    encode_core = model.encode_core(encode_pre)
    preds = logits_to_probabilities(model.encode_post(encode_core), model.classification_activation)

    assert len(preds) == len(sentences)
    assert isinstance(preds, list)
    assert isinstance(preds[0], list)
    assert isinstance(preds[0][0], dict)
    assert isinstance(preds[0][0]["label"], str)
    assert isinstance(preds[0][0]["score"], float)
    assert preds[0][0]["label"] == "admiration"
    assert 0.98 > preds[0][0]["score"] > 0.93

    preds_orig = pipe(sentences, top_k=None, truncation=True)

    assert len(preds_orig) == len(preds)

    for pred_orig, pred in zip(preds_orig, preds):
        assert len(pred_orig) == len(pred)
        # the ONNX repo ships an int8 model: the top class must agree with the fp32
        # pipeline, the tail classes only within a tolerance (their order depends on
        # the int8 kernels of the CPU)
        assert pred_orig[0]["label"] == pred[0]["label"]
        scores = {p["label"]: p["score"] for p in pred}
        for pred_orig_i in pred_orig[:5]:
            assert abs(pred_orig_i["score"] - scores[pred_orig_i["label"]]) < 0.05
