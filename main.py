"""
main.py
-------
نقطة التشغيل الرئيسية.
يربط VMCController + VnyanAgent ويشغّل الـ loop.
"""

import asyncio
import math
import time
import threading
import os

from VMCcontroller import VMCController   # الكود اللي عندك
from Agent import VnyanAgent


# ══════════════════════════════════════════════
# 1. إعدادات
# ══════════════════════════════════════════════

VMC_IP       = "127.0.0.1"
VMC_PORT     = 8000
VRM_PATH     = r"C:\Users\Void\Documents\AvatarSample_I.vrm"   # ← عدّله


# ══════════════════════════════════════════════
# 2. Animation Loop — في thread منفصل
# ══════════════════════════════════════════════

def animation_loop(vmc: VMCController, stop_event: threading.Event):
    """
    يشغّل idle animation مستمرة للـ avatar.
    يتوقف لما stop_event يُضبط.
    """
    tick = 0.0
    while not stop_event.is_set():
        tick += 0.02
        vmc.smooth_move(
            bone_name="Head",
            target_angles={
                "pitch": math.sin(tick * 0.7) * 4.0,
                "yaw"  : math.sin(tick * 0.3) * 6.0,
                "roll" : math.cos(tick * 0.5) * 3.0,
            },
            lerp_speed=0.05,
        )
        # blink كل ~4 ثواني
        blink_phase = math.sin(tick * 0.8)
        if blink_phase > 0.97:
            vmc.smooth_expression({"blink": 100}, normalize=True, smoothness=0.3)
        else:
            vmc.smooth_expression({"blink": 0}, normalize=True, smoothness=0.1)

        time.sleep(1 / 60)

    print("[Animation] loop stopped.")


# ══════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════

def main():
    print("=== vnyan AI Agent ===\n")

    # ── VMC ──
    vmc = VMCController(VMC_IP, VMC_PORT, VRM_PATH)
    print(f"[VMC] Connected to {VMC_IP}:{VMC_PORT}")
    print(f"[VMC] Expressions available: {vmc.available_expressions}\n")

    # ── Agent ──
    agent = VnyanAgent(
        vmc=vmc,
        session_id="session_001",
        model="qwen3:14b",
        temperature=0.75,
    )
    print("[Agent] Ready.\n")

    # ── Animation thread ──
    stop_event = threading.Event()
    anim_thread = threading.Thread(
        target=animation_loop,
        args=(vmc, stop_event),
        daemon=True,
    )
    anim_thread.start()
    print("[Animation] Idle loop started.\n")

    # ── Chat loop ──
    print("اكتب رسالتك (أو 'exit' للخروج، 'save' لإنهاء الجلسة):\n")
    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                break

            if user_input.lower() == "save":
                agent.end_session()
                print("[Session saved]\n")
                continue

            if user_input.lower().startswith("recall:"):
                query = user_input[7:].strip()
                results = agent.recall(query)
                print("[Memory recall]:")
                for i, r in enumerate(results, 1):
                    print(f"  {i}. {r}")
                print()
                continue

            # ← الرد الرئيسي
            print("Agent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response, "\n")

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        stop_event.set()
        agent.end_session()
        print("[Goodbye]")


if __name__ == "__main__":
    main()