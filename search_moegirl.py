#!/usr/bin/env python3
"""
萌娘百科数据搜索工具
从本地 HuggingFace datasets 缓存中搜索萌娘百科条目
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 设置缓存目录
os.environ["HF_HOME"] = "/data0/tcmofashi/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data0/tcmofashi/cache/transformers"


def load_moegirl_dataset():
    """加载萌娘百科数据集"""
    try:
        from datasets import load_dataset
        print("⏳ 正在加载萌娘百科数据集...")
        dataset = load_dataset(
            "KomeijiForce/moe_girl_wiki",
            cache_dir="/data0/tcmofashi/cache/datasets"
        )
        print(f"✅ 数据集加载成功！")
        print(f"   训练集大小: {len(dataset['train'])} 条")
        return dataset
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return None


def search_by_keyword(dataset, keyword: str, field: str = "content", max_results: int = 10) -> List[Dict[str, Any]]:
    """通过关键词搜索数据集"""
    if dataset is None:
        return []

    print(f"\n🔍 搜索关键词: '{keyword}' (字段: {field})")
    results = []

    train_data = dataset["train"]

    # 遍历数据集搜索关键词
    for idx, item in enumerate(train_data):
        if len(results) >= max_results:
            break

        # 获取搜索字段的值
        search_value = item.get(field, "")

        # 检查是否包含关键词
        if keyword.lower() in str(search_value).lower():
            # 从 path 中提取标题
            path = item.get("path", "")
            title = path.split("/")[-1].replace(".txt", "") if "/" in path else path

            results.append({
                "index": idx,
                "title": title,
                "path": path,
                "text": str(search_value)[:500] + "..." if len(str(search_value)) > 500 else str(search_value),
            })

    return results


def search_by_title(dataset, title_keyword: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """通过标题关键词搜索（在path字段中搜索）"""
    if dataset is None:
        return []

    print(f"\n🔍 搜索标题关键词: '{title_keyword}'")
    results = []

    train_data = dataset["train"]

    for idx, item in enumerate(train_data):
        if len(results) >= max_results:
            break

        # 从 path 中提取标题
        path = item.get("path", "")
        title = path.split("/")[-1].replace(".txt", "") if "/" in path else path

        # 检查标题是否包含关键词
        if title_keyword.lower() in title.lower():
            results.append({
                "index": idx,
                "title": title,
                "path": path,
                "text": item.get("content", "")[:500] + "..." if len(item.get("content", "")) > 500 else item.get("content", ""),
            })

    return results


def print_results(results: List[Dict[str, Any]]):
    """打印搜索结果"""
    if not results:
        print("❌ 未找到相关条目")
        return

    print(f"\n✅ 找到 {len(results)} 个相关条目:\n")

    for i, result in enumerate(results, 1):
        print("=" * 80)
        print(f"[{i}] 索引: {result['index']}")
        print(f"    标题: {result['title']}")
        print(f"    路径: {result['path']}")
        print(f"    预览: {result['text'][:200]}...")
        print()


def get_full_entry(dataset, index: int) -> Dict[str, Any]:
    """获取完整条目"""
    train_data = dataset["train"]
    if 0 <= index < len(train_data):
        return train_data[index]
    return None


def interactive_search(dataset):
    """交互式搜索"""
    print("\n" + "=" * 80)
    print("🔍 进入交互搜索模式")
    print("=" * 80)
    print("命令:")
    print("  /title <关键词>  - 按标题搜索")
    print("  /text <关键词>   - 按正文搜索")
    print("  /get <索引>      - 查看完整条目")
    print("  /quit 或 /q      - 退出")
    print()

    while True:
        try:
            user_input = input("🔹 请输入命令: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/q", "quit", "exit"]:
                print("👋 退出搜索模式")
                break

            # 解析命令
            if user_input.startswith("/title "):
                keyword = user_input[7:].strip()
                results = search_by_title(dataset, keyword, max_results=20)
                print_results(results)

            elif user_input.startswith("/text "):
                keyword = user_input[6:].strip()
                results = search_by_keyword(dataset, keyword, field="text", max_results=20)
                print_results(results)

            elif user_input.startswith("/get "):
                try:
                    index = int(user_input[5:].strip())
                    entry = get_full_entry(dataset, index)
                    if entry:
                        print("\n" + "=" * 80)
                        print(f"📄 完整条目 (索引: {index})")
                        print("=" * 80)
                        path = entry.get("path", "")
                        title = path.split("/")[-1].replace(".txt", "") if "/" in path else path
                        print(f"标题: {title}")
                        print(f"路径: {path}")
                        print(f"\n正文:")
                        print(entry.get("content", "N/A"))
                        print()
                    else:
                        print(f"❌ 无效的索引: {index}")
                except ValueError:
                    print("❌ 请输入有效的索引数字")

            else:
                print("❌ 未知命令，请使用 /title, /text, /get 或 /quit")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出搜索模式")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="萌娘百科数据搜索工具")
    parser.add_argument("--keyword", "-k", type=str, help="搜索关键词")
    parser.add_argument("--title", "-t", type=str, help="标题关键词")
    parser.add_argument("--field", "-f", type=str, default="content", choices=["path", "content"], help="搜索字段")
    parser.add_argument("--max-results", "-n", type=int, default=10, help="最大结果数")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")
    parser.add_argument("--save", "-s", type=str, help="保存结果到文件")

    args = parser.parse_args()

    # 加载数据集
    dataset = load_moegirl_dataset()

    if dataset is None:
        return

    # 如果指定了关键词，直接搜索
    if args.keyword:
        results = search_by_keyword(dataset, args.keyword, field=args.field, max_results=args.max_results)
        print_results(results)

        if args.save:
            output_file = Path(args.save)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_file}")

    elif args.title:
        results = search_by_title(dataset, args.title, max_results=args.max_results)
        print_results(results)

        if args.save:
            output_file = Path(args.save)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_file}")

    # 如果没有指定搜索词或进入交互模式
    elif args.interactive or (not args.keyword and not args.title):
        interactive_search(dataset)


if __name__ == "__main__":
    main()
