#!/usr/bin/env python3
"""
模型对比脚本 - 同时测试基础模型和CPT微调模型
用法:
    python compare_models.py "你的问题"
    python compare_models.py --interactive  # 交互模式
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# 设置缓存目录（与 run_cpt.sh 保持一致）
os.environ["HF_HOME"] = "/data0/tcmofashi/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data0/tcmofashi/cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/data0/tcmofashi/cache/datasets"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


class ModelComparator:
    def __init__(
        self,
        base_model_path: str,
        finetuned_model_path: str,
        device_map: str = "auto",
        cache_dir: str = "/data0/tcmofashi/cache/transformers",
    ):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device_map = device_map
        self.cache_dir = cache_dir

        print("=" * 80)
        print("🚀 初始化模型对比器")
        print("=" * 80)
        print(f"📦 基础模型: {base_model_path}")
        print(f"📦 微调模型: {finetuned_model_path}")
        print(f"💾 模型缓存目录: {self.cache_dir}")
        print("=" * 80)

    def _load_model(self, model_path: str) -> tuple:
        """加载模型和tokenizer"""
        print(f"\n⏳ 正在加载模型: {model_path}")
        start_time = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=self.cache_dir)

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device_map,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

            load_time = time.time() - start_time
            print(f"✅ 模型加载完成! (耗时: {load_time:.2f}秒)")

            return model, tokenizer

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise

    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        system_prompt: str = "你是一个回答问题的专家，擅长提供准确、详细、有见地的回答。",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stream: bool = True,
    ) -> str:
        """生成模型回复（支持流式输出）"""
        # 构建 Qwen 格式的消息
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        # 使用 chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 记录生成时间
        start_time = time.time()
        generated_text = ""

        if stream:
            # 流式生成
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "repetition_penalty": 1.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "streamer": streamer,
            }

            # 在新线程中启动生成
            from threading import Thread

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # 实时输出
            for new_text in streamer:
                print(new_text, end="", flush=True)
                generated_text += new_text

            thread.join()
            print()  # 换行
        else:
            # 非流式生成（原有逻辑）
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            print(generated_text)

        gen_time = time.time() - start_time
        return generated_text, gen_time

    def compare(
        self,
        prompt: str,
        system_prompt: str = "你是一个回答问题的专家，擅长提供准确、详细、有见地的回答。",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        save_to_file: bool = True,
        stream: bool = True,
    ):
        """对比两个模型的输出"""
        print("\n" + "=" * 80)
        print("📝 System Prompt")
        print("=" * 80)
        print(f"{system_prompt}\n")

        print("=" * 80)
        print("📝 User Prompt")
        print("=" * 80)
        print(f"{prompt}\n")

        results = {}

        # 测试基础模型
        print("=" * 80)
        print("🔵 基础模型生成中...")
        print("=" * 80)

        base_model, base_tokenizer = self._load_model(self.base_model_path)
        base_response, base_time = self.generate_response(
            base_model,
            base_tokenizer,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
        )

        # 流式模式下不重复打印，非流式模式下打印完整响应
        if not stream:
            print(f"\n{base_response}")

        print(f"\n⏱️  生成时间: {base_time:.2f}秒")
        print(f"📊 生成token数: {len(base_tokenizer.encode(base_response))}")
        print(f"📝 生成状态: {'流式输出' if stream else '非流式输出'}")

        results["base"] = {
            "response": base_response,
            "time": base_time,
            "tokens": len(base_tokenizer.encode(base_response)),
        }

        # 释放显存
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print("🟠 微调模型生成中...")
        print("=" * 80)

        finetuned_model, finetuned_tokenizer = self._load_model(self.finetuned_model_path)
        finetuned_response, finetuned_time = self.generate_response(
            finetuned_model,
            finetuned_tokenizer,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
        )

        # 流式模式下不重复打印，非流式模式下打印完整响应
        if not stream:
            print(f"\n{finetuned_response}")

        print(f"\n⏱️  生成时间: {finetuned_time:.2f}秒")
        print(f"📊 生成token数: {len(finetuned_tokenizer.encode(finetuned_response))}")
        print(f"📝 生成状态: {'流式输出' if stream else '非流式输出'}")

        results["finetuned"] = {
            "response": finetuned_response,
            "time": finetuned_time,
            "tokens": len(finetuned_tokenizer.encode(finetuned_response)),
        }

        # 对比总结
        print("\n" + "=" * 80)
        print("📊 对比总结")
        print("=" * 80)
        print(f"基础模型时间:     {results['base']['time']:.2f}秒")
        print(f"微调模型时间:     {results['finetuned']['time']:.2f}秒")
        print(f"基础模型tokens:   {results['base']['tokens']}")
        print(f"微调模型tokens:   {results['finetuned']['tokens']}")

        # 保存结果
        if save_to_file:
            self._save_results(prompt, results)

        return results

    def _save_results(self, prompt: str, results: Dict[str, Any], system_prompt: str = ""):
        """保存对比结果到文件"""
        output_dir = Path("/data0/tcmofashi/model_comparison_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_{timestamp}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("模型对比结果\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"基础模型: {self.base_model_path}\n")
            f.write(f"微调模型: {self.finetuned_model_path}\n\n")

            f.write("=" * 80 + "\n")
            f.write("System Prompt\n")
            f.write("=" * 80 + "\n")
            f.write(f"{system_prompt}\n\n")

            f.write("=" * 80 + "\n")
            f.write("User Prompt\n")
            f.write("=" * 80 + "\n")
            f.write(f"{prompt}\n\n")

            f.write("=" * 80 + "\n")
            f.write("基础模型回复\n")
            f.write("=" * 80 + "\n")
            f.write(f"{results['base']['response']}\n\n")
            f.write(f"⏱️  生成时间: {results['base']['time']:.2f}秒\n")
            f.write(f"📊 Token数: {results['base']['tokens']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("微调模型回复\n")
            f.write("=" * 80 + "\n")
            f.write(f"{results['finetuned']['response']}\n\n")
            f.write(f"⏱️  生成时间: {results['finetuned']['time']:.2f}秒\n")
            f.write(f"📊 Token数: {results['finetuned']['tokens']}\n")

        print(f"\n💾 结果已保存到: {output_file}")


def interactive_mode(comparator: ModelComparator):
    """交互式模式"""
    print("\n" + "=" * 80)
    print("🎯 进入交互模式 (输入 'quit' 或 'exit' 退出)")
    print("=" * 80)

    # 询问是否使用自定义 system prompt
    use_custom_system = input("\n是否使用自定义 system prompt? (y/N): ").strip().lower()

    default_system_prompt = "你是一个回答问题的专家，擅长提供准确、详细、有见地的回答。"
    system_prompt = default_system_prompt

    if use_custom_system == "y":
        custom_prompt = input("请输入 system prompt (留空使用默认): ").strip()
        if custom_prompt:
            system_prompt = custom_prompt

    print(f"\n📋 System Prompt: {system_prompt}")
    print("=" * 80 + "\n")

    while True:
        try:
            prompt = input("🔹 请输入你的问题: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ["quit", "exit", "q"]:
                print("\n👋 退出交互模式")
                break

            comparator.compare(prompt, system_prompt=system_prompt, save_to_file=True)

            print("\n" * 2)

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出交互模式")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="模型对比工具")
    parser.add_argument("prompt", nargs="?", help="要测试的prompt（不提供则进入交互模式）")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="基础模型路径")
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="/data0/tcmofashi/saves/qwen3-30b/cpt/h20_liger_fast",
        help="微调模型路径",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大生成token数")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    parser.add_argument("--no-save", action="store_true", help="不保存结果到文件")
    parser.add_argument("--cache-dir", type=str, default="/data0/tcmofashi/cache/transformers", help="模型缓存目录")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="你是一个回答问题的专家，擅长提供准确、详细、有见地的回答。",
        help="System prompt",
    )

    args = parser.parse_args()

    # 创建对比器
    comparator = ModelComparator(
        base_model_path=args.base_model, finetuned_model_path=args.finetuned_model, cache_dir=args.cache_dir
    )

    # 判断运行模式
    if args.interactive or not args.prompt:
        interactive_mode(comparator)
    else:
        comparator.compare(
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            save_to_file=not args.no_save,
            stream=not args.no_stream,
        )


if __name__ == "__main__":
    main()
