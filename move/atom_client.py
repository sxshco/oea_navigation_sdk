# On Server

import asyncio
import json
import threading
import time
from pynput.keyboard import Listener, Key
import websockets

# 使用枚举或字典来管理常量，提高可读性
class ApiId:
    DAMP = 1001
    BALANCESTAND = 1002
    STOPMOVE = 1003
    STANDUP = 1004
    SIT = 1005
    MOVE = 1008

class Action:
    FORWARD = "[ACT_FORWARD]"
    LEFT = "[ACT_LEFT]"
    RIGHT = "[ACT_RIGHT]"
    STOP = "[ACT_STOP]" # 注意：这对应原代码的 'S' 键，即暂停0.5秒


# ================================================================
#  核心动作控制器 (Core Action Controller)
# ================================================================

class UnitreeActionController:
    """
    负责与Unitree机器人进行WebSocket通信，并执行原子动作。
    这个类是平台无关的，可以被任何上层逻辑（如键盘、大模型API）调用。
    """
    def __init__(self, server_host='0.0.0.0', server_port=8000):
        self.host = server_host
        self.port = server_port
        self.websocket = None
        self.running = True
        self.command_lock = asyncio.Lock() # 在异步环境中使用asyncio.Lock
        self.loop = None

        # 动作参数定义
        self.motion_params = {
            Action.FORWARD: {"x": 0.55, "y": 0.0, "z": -0.01},
            Action.LEFT:    {"x": 0.0, "y": 0.0, "z": 0.95}, # 左转
            Action.RIGHT:   {"x": 0.0, "y": 0.0, "z": -0.95},# 右转
        }
        self.action_duration = 0.5 # 每个动作的持续时间

    async def _send_websocket_message(self, api_id, parameter=None):
        """底层的WebSocket消息发送函数"""
        if not self.websocket or not self.running:
            print(f"指令 [{api_id}] 未发送: 连接不可用。")
            return False
        
        cmd = {"api_id": api_id, "parameter": parameter or {}}
        try:
            await self.websocket.send(json.dumps(cmd))
            return True
        except websockets.exceptions.ConnectionClosed:
            print(f"指令 [{api_id}] 未发送: WebSocket连接已关闭。")
            self.websocket = None
            return False
        except Exception as e:
            print(f"WebSocket通信错误: {e}")
            self.websocket = None
            return False

    async def _perform_atomic_move(self, action_string):
        """执行一个持续性的移动动作（前进、左转、右转）"""
        params = self.motion_params.get(action_string)
        if not params:
            print(f"未知动作: {action_string}")
            return

        print(f"执行动作: {action_string}")
        if await self._send_websocket_message(ApiId.MOVE, parameter=params):
            await asyncio.sleep(self.action_duration)
            await self._send_websocket_message(ApiId.STOPMOVE)
    
    async def _perform_stop(self):
        """执行一个短暂的停止动作"""
        print(f"执行动作: {Action.STOP}")
        if await self._send_websocket_message(ApiId.STOPMOVE):
            await asyncio.sleep(self.action_duration)

    async def execute_action(self, action_string: str):
        """
        【核心接口】供外部调用，执行指定的原子动作。
        参数:
            action_string (str): 必须是 Action 类中定义的字符串之一。
        """
        if not self.websocket:
            print("无法执行动作：未连接到机器人。")
            return

        # 使用异步锁确保同一时间只有一个动作在执行
        async with self.command_lock:
            if action_string in self.motion_params:
                await self._perform_atomic_move(action_string)
            elif action_string == Action.STOP:
                await self._perform_stop()
            else:
                print(f"警告：接收到未知的动作指令 '{action_string}'")

    # --- WebSocket服务器和生命周期管理 ---
    async def handler(self, websocket):
        """处理新的WebSocket连接"""
        print(f"机器人已连接: {websocket.remote_address}")
        self.websocket = websocket
        
        await self.send_initial_commands_async()

        try:
            # 保持连接，可以用于接收机器人状态更新（如果需要）
            async for message in websocket:
                print(f"收到来自机器人的消息: {message}")
        except websockets.exceptions.ConnectionClosed:
            print(f"机器人断开连接: {websocket.remote_address}")
        finally:
            self.websocket = None
            print("机器人连接已关闭。")

    async def send_initial_commands_async(self):
        print("初始化: 发送站立和平衡站立指令...")
        if await self._send_websocket_message(ApiId.STANDUP):
            await asyncio.sleep(2)
        if await self._send_websocket_message(ApiId.BALANCESTAND):
            await asyncio.sleep(1)
        print("初始化完成。等待您的指令...")

    async def send_exit_commands_async(self):
        print("\n程序退出，发送停止、坐下和阻尼指令...")
        if await self._send_websocket_message(ApiId.STOPMOVE):
            await asyncio.sleep(1)
        if await self._send_websocket_message(ApiId.SIT):
            await asyncio.sleep(3)
        await self._send_websocket_message(ApiId.DAMP)
        print("退出指令发送完成。")
    
    async def start_server(self):
        """启动WebSocket服务器并永远运行"""
        print(f"===== Unitree 动作控制器 =====\n等待机器人连接到: ws://{self.host}:{self.port}")
        self.loop = asyncio.get_running_loop()
        async with websockets.serve(self.handler, self.host, self.port):
            # 保持服务器运行，直到 self.running 变为 False
            while self.running:
                await asyncio.sleep(0.1)

    async def stop_server(self):
        """
        这是一个优雅的关闭流程：
        1. 首先发送退出指令。
        2. 然后再设置 self.running = False 来通知服务器主循环退出。
        """
        if self.running:
            # 1. 先执行需要网络连接的操作
            await self.send_exit_commands_async()
            
            # 2. 然后再改变状态，触发服务器关闭
            self.running = False
            print("服务器正在关闭...")


# ================================================================
#  键盘控制接口 (Keyboard Control Interface)
# ================================================================

class KeyboardInterface:
    """
    负责将键盘输入转换为动作指令，并调用UnitreeActionController。
    """
    def __init__(self, controller: UnitreeActionController):
        self.controller = controller
        self.key_map = {
            'w': Action.FORWARD,
            'a': Action.LEFT,
            'd': Action.RIGHT,
            's': Action.STOP,
        }
        self.listener_thread = threading.Thread(target=self._start_keyboard_listener, daemon=True)

        print("\n===== 键盘控制器已激活 =====")
        print("控制按键:")
        for key, action in self.key_map.items():
            print(f"  {key.upper()}: {action}")
        print("  ESC: 退出程序")
        print("============================\n")

    def start(self):
        self.listener_thread.start()

    def on_press(self, key):
        if not self.controller.running:
            return False # 停止监听器

        if key == Key.esc:
            print("程序退出指令接收。")
            if self.controller.loop and self.controller.running:
                # 【核心修正】调用 stop_server，它会处理好顺序
                asyncio.run_coroutine_threadsafe(self.controller.stop_server(), self.controller.loop)
            return False # 直接返回False，让监听器线程结束
        
        try:
            key_char = key.char.lower()
            action = self.key_map.get(key_char)
            if action and self.controller.loop:
                # 检查锁的状态，避免在动作执行时发送新指令
                if not self.controller.command_lock.locked():
                    asyncio.run_coroutine_threadsafe(
                        self.controller.execute_action(action),
                        self.controller.loop
                    )
                else:
                    print(">> 有其他指令正在执行中，此按键已被忽略。")

        except AttributeError:
            # 按下了非字符键
            pass

    def _start_keyboard_listener(self):
        with Listener(on_press=self.on_press) as listener:
            listener.join()
        print("键盘监听器已停止。")


# ================================================================
#  主程序入口 (Main Execution)
# ================================================================

async def main():
    controller = UnitreeActionController(server_port=8000)
    
    # 启动键盘接口
    keyboard_control = KeyboardInterface(controller)
    keyboard_control.start()

    # 在这里，您可以启动您的大模型API服务，并让它也调用 controller.execute_action()
    # 例如:
    # my_llm_service = MyLargeLanguageModelService(controller)
    # await my_llm_service.start()

    # 启动并运行WebSocket服务器，直到被键盘的ESC键停止
    await controller.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        print("程序已完全退出。")