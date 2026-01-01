#!/usr/bin/env python3
"""
Provider负载均衡器
支持加权（按RPM）和轮询策略
"""

import random
from typing import List, Dict, Optional
from collections import deque


class LoadBalancer:
    """Provider负载均衡器"""

    def __init__(self, strategy: str = "weighted"):
        """
        初始化负载均衡器

        Args:
            strategy: 负载均衡策略
                - "weighted": 按RPM权重分配
                - "round_robin": 轮询
                - "random": 随机选择
        """
        self.strategy = strategy
        self.providers = {}  # {provider_key: {"rpm": int, "weight": float}}
        self.total_rpm = 0
        self.round_robin_queue = deque()

    def add_provider(self, provider_key: str, rpm: int):
        """添加provider"""
        self.providers[provider_key] = {
            "rpm": rpm,
            "weight": 0.0  # 将在update_weights中计算
        }
        self.round_robin_queue.append(provider_key)
        self.update_weights()

    def update_weights(self):
        """更新权重（基于RPM）"""
        self.total_rpm = sum(p["rpm"] for p in self.providers.values())

        # 计算每个provider的权重
        for provider_key in self.providers:
            rpm = self.providers[provider_key]["rpm"]
            if self.total_rpm > 0:
                self.providers[provider_key]["weight"] = rpm / self.total_rpm
            else:
                self.providers[provider_key]["weight"] = 1.0 / len(self.providers)

    def get_provider(self) -> Optional[str]:
        """根据策略选择一个provider"""
        if not self.providers:
            return None

        if self.strategy == "weighted":
            return self._weighted_select()
        elif self.strategy == "round_robin":
            return self._round_robin_select()
        elif self.strategy == "random":
            return random.choice(list(self.providers.keys()))
        else:
            # 默认使用加权
            return self._weighted_select()

    def _weighted_select(self) -> str:
        """按RPM权重选择provider"""
        # 生成0-1之间的随机数
        rand = random.random()

        # 根据权重选择provider
        cumulative = 0.0
        for provider_key, provider_info in self.providers.items():
            cumulative += provider_info["weight"]
            if rand <= cumulative:
                return provider_key

        # 如果由于浮点精度问题没选中，返回最后一个
        return list(self.providers.keys())[-1]

    def _round_robin_select(self) -> str:
        """轮询选择provider"""
        if not self.round_robin_queue:
            return list(self.providers.keys())[0]

        # 取出队列头部元素并放到尾部
        provider_key = self.round_robin_queue.popleft()
        self.round_robin_queue.append(provider_key)
        return provider_key

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "strategy": self.strategy,
            "total_providers": len(self.providers),
            "total_rpm": self.total_rpm,
            "providers": {
                key: {
                    "rpm": info["rpm"],
                    "weight": info["weight"],
                    "percentage": info["weight"] * 100
                }
                for key, info in self.providers.items()
            }
        }

    def print_status(self):
        """打印当前状态（用于调试）"""
        status = self.get_status()
        print(f"\n📊 负载均衡状态 (策略: {status['strategy']})")
        print(f"   总RPM: {status['total_rpm']}")
        print(f"   Providers:")
        for key, info in status['providers'].items():
            print(f"   - {key}: RPM={info['rpm']}, 权重={info['weight']:.2%} ({info['percentage']:.1f}%)")
