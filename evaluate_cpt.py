#!/usr/bin/env python3
"""
增量预训练效果评估脚本
评估基础模型与微调模型在特定知识领域的事实准确性和幻觉率
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


# 设置缓存目录
os.environ["HF_HOME"] = "/data0/tcmofashi/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data0/tcmofashi/cache/transformers"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


class CPTEvaluator:
    """增量预训练效果评估器"""

    def __init__(
        self,
        base_model_path: str,
        finetuned_model_path: str,
        cache_dir: str = "/data0/tcmofashi/cache/transformers",
    ):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.cache_dir = cache_dir

    def _load_model(self, model_path: str):
        """加载模型和tokenizer"""
        print(f"⏳ 加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        return model, tokenizer

    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """生成模型回复"""
        messages = [
            {"role": "system", "content": "你是一个回答问题的专家，擅长提供准确、详细、有见地的回答。"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def evaluate_facts(
        self,
        response: str,
        key_facts: List[str],
        false_facts: List[str],
    ) -> Dict[str, Any]:
        """评估单个回答的事实准确性"""
        results = {
            "key_facts_mentioned": [],
            "key_facts_missed": [],
            "false_facts_mentioned": [],
            "hallucination_count": 0,
        }

        for fact in key_facts:
            if fact in response:
                results["key_facts_mentioned"].append(fact)
            else:
                results["key_facts_missed"].append(fact)

        for fact in false_facts:
            if fact in response:
                results["false_facts_mentioned"].append(fact)
                results["hallucination_count"] += 1

        return results

    def calculate_perplexity(self, model, tokenizer, texts: List[str]) -> float:
        """计算模型在给定文本上的困惑度"""
        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def run_evaluation(
        self,
        test_cases: List[Dict[str, Any]],
        output_dir: str = "/data0/tcmofashi/evaluation_results",
    ) -> Dict[str, Any]:
        """运行完整评估"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "base_model": {},
            "finetuned_model": {},
            "comparison": {},
        }

        # 测试基础模型
        print("\n" + "=" * 80)
        print("🔵 评估基础模型")
        print("=" * 80)
        base_model, base_tokenizer = self._load_model(self.base_model_path)
        results["base_model"] = self._evaluate_model(
            base_model,
            base_tokenizer,
            test_cases,
            "base_model",
        )
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()

        # 测试微调模型
        print("\n" + "=" * 80)
        print("🟠 评估微调模型")
        print("=" * 80)
        finetuned_model, finetuned_tokenizer = self._load_model(self.finetuned_model_path)
        results["finetuned_model"] = self._evaluate_model(
            finetuned_model,
            finetuned_tokenizer,
            test_cases,
            "finetuned_model",
        )

        # 对比分析
        results["comparison"] = self._compare_results(
            results["base_model"],
            results["finetuned_model"],
        )

        # 保存结果
        timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"evaluation_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 评估结果已保存到: {output_file}")
        self._print_summary(results)

        return results

    def _evaluate_model(
        self,
        model,
        tokenizer,
        test_cases: List[Dict[str, Any]],
        model_name: str,
    ) -> Dict[str, Any]:
        """评估单个模型"""
        model_results = {
            "responses": [],
            "metrics": {
                "total_questions": len(test_cases),
                "total_key_facts": sum(len(c.get("key_facts", [])) for c in test_cases),
                "key_facts_recalled": 0,
                "false_facts_mentioned": 0,
                "hallucination_rate": 0.0,
            },
        }

        for i, case in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] 问题: {case['question'][:50]}...")

            response = self.generate_response(model, tokenizer, case["question"])
            eval_result = self.evaluate_facts(
                response,
                case.get("key_facts", []),
                case.get("false_facts", []),
            )

            model_results["responses"].append({
                "question": case["question"],
                "response": response,
                "evaluation": eval_result,
            })

            model_results["metrics"]["key_facts_recalled"] += len(eval_result["key_facts_mentioned"])
            model_results["metrics"]["false_facts_mentioned"] += len(eval_result["false_facts_mentioned"])

        # 计算召回率和幻觉率
        metrics = model_results["metrics"]
        metrics["recall_rate"] = (
            metrics["key_facts_recalled"] / metrics["total_key_facts"]
            if metrics["total_key_facts"] > 0 else 0.0
        )
        metrics["hallucination_rate"] = (
            metrics["false_facts_mentioned"] / len(test_cases)
            if len(test_cases) > 0 else 0.0
        )

        return model_results

    def _compare_results(
        self,
        base_results: Dict[str, Any],
        finetuned_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """对比两个模型的结果"""
        comparison = {
            "base_model_metrics": base_results["metrics"],
            "finetuned_model_metrics": finetuned_results["metrics"],
            "improvements": {},
        }

        base_metrics = base_results["metrics"]
        fine_metrics = finetuned_results["metrics"]

        comparison["improvements"] = {
            "recall_rate": {
                "base": base_metrics["recall_rate"],
                "finetuned": fine_metrics["recall_rate"],
                "absolute_change": fine_metrics["recall_rate"] - base_metrics["recall_rate"],
                "relative_change": (
                    (fine_metrics["recall_rate"] - base_metrics["recall_rate"]) / base_metrics["recall_rate"] * 100
                    if base_metrics["recall_rate"] > 0 else 0.0
                ),
            },
            "hallucination_rate": {
                "base": base_metrics["hallucination_rate"],
                "finetuned": fine_metrics["hallucination_rate"],
                "absolute_change": fine_metrics["hallucination_rate"] - base_metrics["hallucination_rate"],
                "relative_change": (
                    (base_metrics["hallucination_rate"] - fine_metrics["hallucination_rate"]) / base_metrics["hallucination_rate"] * 100
                    if base_metrics["hallucination_rate"] > 0 else 0.0
                ),
            },
        }

        return comparison

    def _print_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "=" * 80)
        print("📊 评估结果摘要")
        print("=" * 80)

        comp = results["comparison"]["improvements"]

        print("\n📈 事实召回率")
        print(f"  基础模型: {comp['recall_rate']['base']:.2%}")
        print(f"  微调模型: {comp['recall_rate']['finetuned']:.2%}")
        print(f"  绝对提升: {comp['recall_rate']['absolute_change']:+.2%}")
        print(f"  相对提升: {comp['recall_rate']['relative_change']:+.1f}%")

        print("\n⚠️ 幻觉率")
        print(f"  基础模型: {comp['hallucination_rate']['base']:.2%}")
        print(f"  微调模型: {comp['hallucination_rate']['finetuned']:.2%}")
        print(f"  绝对变化: {comp['hallucination_rate']['absolute_change']:+.2%}")
        print(f"  相对改善: {comp['hallucination_rate']['relative_change']:+.1f}%")


# 示例测试用例
SAMPLE_TEST_CASES = [
    {
        "question": "为什么大家称呼种崎敦美为华哥？",
        "key_facts": ["桐谷华", "Galgame", "艺名", "马甲"],
        "false_facts": ["艾米莉娅", "伏黑惠", "谐音", "发音"],
        "category": "声优知识"
    },
    {
        "question": "长谷川育美的真实生日和出生地是哪里？她有哪些代表作？",
        "key_facts": ["5月31日", "栃木县", "86", "芙拉蒂蕾娜", "弱势角色友崎君"],
        "false_facts": ["10月12日", "埼玉县", "四宫辉夜", "北条加莲", "12月12日", "神奈川县"],
        "category": "声优知识"
    },
    {
        "question": "四宫辉夜的声优是谁？",
        "key_facts": ["古贺葵"],
        "false_facts": ["长谷川育美", "种崎敦美", "佐仓绫音"],
        "category": "角色配音"
    },
]


def main():
    parser = argparse.ArgumentParser(description="增量预训练效果评估")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="基础模型路径"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="/data0/tcmofashi/saves/qwen3-30b/cpt/h20_liger_fast",
        help="微调模型路径"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="测试用例JSON文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data0/tcmofashi/evaluation_results",
        help="结果输出目录"
    )

    args = parser.parse_args()

    # 加载测试用例
    if args.test_file:
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    else:
        test_cases = SAMPLE_TEST_CASES

    # 创建评估器并运行评估
    evaluator = CPTEvaluator(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
    )

    results = evaluator.run_evaluation(
        test_cases=test_cases,
        output_dir=args.output_dir,
    )

    return results


if __name__ == "__main__":
    main()
