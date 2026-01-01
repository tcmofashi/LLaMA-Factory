#!/usr/bin/env python3
"""
测试前3个动画的全流程生成
"""

import sys
import json
import os

sys.path.insert(0, 'react_agent')

from batch_process_all_anime import load_anime_list, process_single_anime

# 配置
ANIME_LIST_FILE = "/home/tcmofashi/LLaMA-Factory/agent_data/anime.txt"
TEST_COUNT = 3

def main():
    print(f"\n{'='*100}")
    print(f"# 测试前{TEST_COUNT}个动画的全流程生成")
    print(f"{'='*100}\n")

    # 加载动画列表
    anime_list = load_anime_list(ANIME_LIST_FILE)
    test_anime = anime_list[:TEST_COUNT]

    print(f"📊 测试动画列表:")
    for i, anime in enumerate(test_anime, 1):
        print(f"   {i}. {anime}")
    print(f"\n{'='*100}\n")

    # 处理每个动画
    results = []
    for i, anime_name in enumerate(test_anime, 1):
        result = process_single_anime(anime_name, i, TEST_COUNT)
        results.append(result)

        # 保存中间结果
        with open("/home/tcmofashi/LLaMA-Factory/agent_data/test_3_progress.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 最终统计
    print(f"\n{'='*100}")
    print(f"📊 测试完成")
    print(f"{'='*100}\n")

    success_count = sum(1 for r in results if r.get("qa_success") and r.get("train_success"))
    failed_count = TEST_COUNT - success_count

    print(f"总计: {TEST_COUNT} 个动画")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {failed_count} 个")
    print(f"📝 生成问题: {sum(r.get('qa_questions', 0) for r in results)} 个")
    print(f"📚 生成训练数据: {sum(r.get('train_questions', 0) for r in results)} 条")

    # 详细结果
    print(f"\n📄 详细结果:")
    for r in results:
        status = "✅" if (r.get("qa_success") and r.get("train_success")) else "❌"
        print(f"   {status} {r['anime']}")
        print(f"      QA: {r.get('qa_questions', 0)}个问题 [{'成功' if r.get('qa_success') else '失败'}]")
        print(f"      训练: {r.get('train_questions', 0)}条数据 [{'成功' if r.get('train_success') else '失败'}]")
        if r.get('error'):
            print(f"      错误: {r['error']}")

    # 保存最终结果
    final_summary = {
        "total": TEST_COUNT,
        "qa_success": sum(1 for r in results if r.get("qa_success")),
        "train_success": sum(1 for r in results if r.get("train_success")),
        "total_questions": sum(r.get('qa_questions', 0) for r in results),
        "total_training_data": sum(r.get('train_questions', 0) for r in results),
        "results": results
    }

    summary_file = "/home/tcmofashi/LLaMA-Factory/agent_data/test_3_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"\n📄 结果已保存到: {summary_file}")

    # 列出失败的动画
    if failed_count > 0:
        print(f"\n❌ 失败的动画:")
        for r in results:
            if not (r.get("qa_success") and r.get("train_success")):
                print(f"   - {r['anime']}: {r.get('error', '未知错误')}")

    return success_count == TEST_COUNT

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
