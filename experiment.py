"""Train a QLoRA adapter or evaluate Qwen2.5-VL on a local manifest."""
import argparse
import json
import re
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def messages(question, answer=None):
    result = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}
    ]}]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


def main(args):
    import torch
    from peft import LoraConfig, PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    start = time.monotonic()
    model_options = {"revision": args.revision, "torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.mode == "train":
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **model_options)
    processor = AutoProcessor.from_pretrained(MODEL, revision=args.revision, max_pixels=512 * 28 * 28)
    if args.mode == "merge":
        if not args.adapter:
            raise ValueError("--adapter is required to merge")
        PeftModel.from_pretrained(model, args.adapter).merge_and_unload().save_pretrained(args.output)
        processor.save_pretrained(args.output)
        return

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    if args.mode == "train":
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        def collate(batch):
            images = [Image.open(x["image"]).convert("RGB") for x in batch]
            full_text = [
                processor.apply_chat_template(messages(x["question"], x["answer"]), tokenize=False)
                for x in batch
            ]
            prompt_text = [
                processor.apply_chat_template(
                    messages(x["question"]), tokenize=False, add_generation_prompt=True
                ) for x in batch
            ]
            full = processor(text=full_text, images=images, padding=True, return_tensors="pt")
            prompt = processor(text=prompt_text, images=images, padding=True, return_tensors="pt")
            labels = full["input_ids"].clone()
            labels[full["attention_mask"] == 0] = -100
            for index, length in enumerate(prompt["attention_mask"].sum(dim=1).tolist()):
                if not torch.equal(prompt["input_ids"][index, :length], full["input_ids"][index, :length]):
                    raise ValueError("Chat template prompt is not a prefix of the training sample")
                labels[index, :length] = -100
            full["labels"] = labels
            return full

        processor.tokenizer.padding_side = "right"
        config = SFTConfig(
            output_dir=args.output, num_train_epochs=args.epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, gradient_checkpointing=True, bf16=True,
            remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
            report_to="none", seed=42,
        )
        trainer = SFTTrainer(
            model=model, args=config, train_dataset=Dataset.from_list(rows), data_collator=collate,
            processing_class=processor,
            peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        trainer.train()
        trainer.save_model(args.output)
        processor.save_pretrained(args.output)
        print(json.dumps({"elapsed_seconds": time.monotonic() - start}))
        return

    from rouge_score import rouge_scorer

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    predictions, exact, rouge = [], [], []
    for item in rows:
        with Image.open(item["image"]) as image:
            prompt = processor.apply_chat_template(
                messages(item["question"]), tokenize=False, add_generation_prompt=True
            )
            batch = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            ids = model.generate(**batch, do_sample=False, max_new_tokens=96)
        prediction = processor.decode(ids[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if item.get("choices"):
            option = re.fullmatch(r"\s*([A-Za-z])[.)]?\s*", prediction)
            exact.append(bool(option and option[1].upper() == item["answer"].upper()))
        else:
            rouge.append(scorer.score(item["answer"], prediction)["rougeL"].fmeasure)
        predictions.append({"id": item["id"], "reference": item["answer"], "prediction": prediction})
    result = {
        "model": MODEL, "revision": args.revision, "adapter": args.adapter, "rows": len(rows),
        "multiple_choice_count": len(exact), "open_answer_count": len(rouge),
        "accuracy": sum(exact) / len(exact) if exact else None,
        "rouge_l": sum(rouge) / len(rouge) if rouge else None,
        "elapsed_seconds": time.monotonic() - start, "predictions": predictions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval", "merge"])
    parser.add_argument("--manifest")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, choices=[1, 2, 3], default=1)
    main(parser.parse_args())
