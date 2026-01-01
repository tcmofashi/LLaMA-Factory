#!/usr/bin/env python3
"""
API速率限制器
支持令牌桶算法，控制API调用频率
"""

import time
from typing import Optional
from collections import deque


class RateLimiter:
    """速率限制器 - 令牌桶算法"""

    def __init__(self, rate: int, period: int = 60):
        """
        初始化速率限制器

        Args:
            rate: 时间周期内允许的请求数量
            period: 时间周期（秒），默认60秒
        """
        self.rate = rate
        self.period = period
        self.tokens = rate  # 当前令牌数
        self.last_update = time.time()
        self.request_times = deque()  # 记录请求时间

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取令牌（阻塞等待）

        Args:
            timeout: 最大等待时间（秒），None表示无限等待

        Returns:
            bool: 是否成功获取令牌
        """
        start_time = time.time()

        while True:
            # 检查是否可以获取令牌
            if self._try_acquire():
                return True

            # 检查超时
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # 等待一段时间再试
            wait_time = self._get_wait_time()
            time.sleep(min(wait_time, 1.0))  # 最多等待1秒

    def _try_acquire(self) -> bool:
        """尝试获取令牌（非阻塞）"""
        current_time = time.time()

        # 更新令牌桶
        elapsed = current_time - self.last_update
        if elapsed >= self.period:
            # 重置令牌桶
            self.tokens = self.rate
            self.request_times.clear()
            self.last_update = current_time
        else:
            # 清理过期的请求记录
            cutoff_time = current_time - self.period
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()
                self.tokens += 1

        # 检查是否有可用令牌
        if self.tokens > 0:
            self.tokens -= 1
            self.request_times.append(current_time)
            return True

        return False

    def _get_wait_time(self) -> float:
        """计算需要等待的时间"""
        if self.request_times:
            # 等到最老的请求过期
            oldest_request = self.request_times[0]
            wait_time = self.period - (time.time() - oldest_request)
            return max(0, wait_time)
        return 0

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "available_tokens": self.tokens,
            "rate": self.rate,
            "period": self.period,
            "recent_requests": len(self.request_times),
        }
