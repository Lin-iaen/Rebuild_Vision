"""GPIO 按键检测模块。

基于 gpiozero 库，提供阻塞式按键读取接口。

电路:
    按键一端接 GPIO，另一端接 GND。
    启用内部上拉 (pull_up=True)，按下时为低电平。

用法::

    from gpio_keys import KeypadController

    keys = KeypadController({5: "enter", 6: "1", 26: "2", 16: "3"})
    while True:
        key = keys.wait_key()
        if key:
            print(f"按下: {key}")
    keys.cleanup()
"""

from __future__ import annotations

from queue import Empty, Queue

from gpiozero import Button


class KeypadController:
    """GPIO 按键检测器。

    参数:
        pin_map: {GPIO编号: "按键名"}, 如 {5: "enter", 6: "1"}
        bounce_time: 消抖时间(秒), 默认 0.05
    """

    def __init__(self, pin_map: dict[int, str], bounce_time: float = 0.05) -> None:
        self._queue: Queue[str] = Queue()
        self._buttons: list[Button] = []

        for pin, name in pin_map.items():
            btn = Button(pin, pull_up=True, bounce_time=bounce_time)
            btn.when_pressed = lambda n=name: self._queue.put(n)
            self._buttons.append(btn)

    def wait_key(self, timeout: float | None = None) -> str | None:
        """阻塞等待按键按下。

        参数:
            timeout: 超时时间(秒)。None 为无限等待。

        返回: 按键名，超时返回 None
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def cleanup(self) -> None:
        """释放所有 GPIO 资源。"""
        for btn in self._buttons:
            btn.close()
