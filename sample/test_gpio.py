"""GPIO 按键测试脚本。

验证硬件接线和按键稳定性。

用法:
    python3 sample/test_gpio.py
    Ctrl+C 退出
"""

import sys
from datetime import datetime

sys.path.insert(0, ".")

from gpio_keys import KeypadController

# ===== 引脚映射 =====
PIN_MAP = {
    5:  "enter",
    6:  "1",
    26: "2",
    16: "3",
    25: "r",
    24: "q",
}


def main():
    print("=" * 50)
    print("GPIO 按键测试")
    print("=" * 50)
    print("按键映射:")
    for pin, name in PIN_MAP.items():
        print(f"  GPIO{pin:<4} → {name}")
    print("-" * 50)
    print("按下任意按键测试，按 q 退出")
    print("-" * 50)

    keys = KeypadController(PIN_MAP, bounce_time=0.05)

    try:
        while True:
            key = keys.wait_key()
            if key is None:
                continue

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            # 找到对应的 GPIO 编号
            pin = next(p for p, n in PIN_MAP.items() if n == key)
            print(f"[{ts}] GPIO{pin:<4} → {key}")

            if key == "q":
                break

    except KeyboardInterrupt:
        print()
    finally:
        keys.cleanup()
        print("已退出")


if __name__ == "__main__":
    main()
