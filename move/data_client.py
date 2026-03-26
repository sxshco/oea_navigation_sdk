#On Server
import cv2
import pyaudio
import socket
import threading
import numpy as np
import struct
from collections import defaultdict
import torch
import torchaudio
import torchaudio.transforms as T
import time

# ================================================================
#  常量定义 (Constants)
# ================================================================
VIDEO_PORT = 5220
AUDIO_PORT = 5221
STATE_PORT = 5222 # 新增：机器人状态端口

# 音频参数（假设动态获取，但保留默认值）
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
DEFAULT_AUDIO_CHANNELS = 2
DEFAULT_AUDIO_RATE = 16000

# 模型输入格式参数
MODEL_IMG_SHAPE = (256, 256)
MODEL_N_MELS = 128
MODEL_N_FFT = 2048
MODEL_HOP_LENGTH = 512
MODEL_AUDIO_BUFFER_SIZE = (44 - 1) * MODEL_HOP_LENGTH + MODEL_N_FFT

# ================================================================
#  1. 核心数据管理与处理类 (DataManager)
# ================================================================

class DataManager:
    """
    负责接收原始数据，将其处理成模型所需的格式，并维护最新状态。
    """
    def __init__(self):
        # 原始数据
        self.latest_rgb_frame = None
        self.latest_audio_chunk = None
        
        # 模型输入数据
        self.model_rgb_input = np.zeros((3, *MODEL_IMG_SHAPE, 3), dtype=np.uint8)
        self.model_audio_input = np.zeros((MODEL_AUDIO_BUFFER_SIZE,), dtype=np.float32)
        # 新增：机器人状态
        self.model_state_input = np.zeros(3, dtype=np.float32) # [x, y, yaw]
        
        # 动态音频参数
        self.audio_rate = None
        self.audio_channels = None

        self.lock = threading.Lock()
        print("DataManager initialized.")

    def set_audio_params(self, rate, channels):
        if self.audio_rate is None:
            with self.lock:
                self.audio_rate = rate; self.audio_channels = channels
                print(f"Audio parameters set: Rate={self.audio_rate}, Channels={self.audio_channels}")

    def update_rgb(self, frame: np.ndarray):
        if frame is None: return
        with self.lock:
            self.latest_rgb_frame = frame.copy()
            resized_frame = cv2.resize(frame, MODEL_IMG_SHAPE, interpolation=cv2.INTER_AREA)
            self.model_rgb_input[0] = resized_frame; self.model_rgb_input[1] = resized_frame; self.model_rgb_input[2] = resized_frame

    def update_audio(self, audio_chunk: bytes):
        if self.audio_channels is None: return
        with self.lock:
            self.latest_audio_chunk = audio_chunk
            raw_samples = np.frombuffer(audio_chunk, dtype=np.int16)
            if raw_samples.size > 0:
                reshaped_samples = raw_samples.reshape(-1, self.audio_channels)
                float_samples = reshaped_samples.astype(np.float32) / 32768.0
                mono_samples = float_samples.mean(axis=1)
                if mono_samples.size > 0:
                    self.model_audio_input = np.roll(self.model_audio_input, -mono_samples.size)
                    self.model_audio_input[-mono_samples.size:] = mono_samples
    
    def update_state(self, state_array: np.ndarray):
        """【新增接口】更新机器人状态"""
        with self.lock:
            self.model_state_input = state_array

    def get_model_input(self):
        """【核心修改】在返回的字典中加入 state 字段"""
        with self.lock:
            rgb_input = self.model_rgb_input.copy()
            state_input = self.model_state_input.copy()
            
            rate = self.audio_rate or DEFAULT_AUDIO_RATE # 使用动态或默认采样率
            
            mel_transform = T.MelSpectrogram(
                sample_rate=rate, n_fft=MODEL_N_FFT, hop_length=MODEL_HOP_LENGTH, n_mels=MODEL_N_MELS, power=2.0
            )
            audio_tensor = torch.tensor(self.model_audio_input, dtype=torch.float32).unsqueeze(0)
            mel_spec = mel_transform(audio_tensor).squeeze(0)
            log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            log_mel_spec_np = log_mel_spec.numpy()
            audio_input = np.stack([log_mel_spec_np, log_mel_spec_np])
            
            return {
                "rgb": rgb_input,
                "audio": audio_input,
                "state": state_input # 新增字段
            }

# ================================================================
#  2. 媒体数据显示类 (MediaDisplayer) - 无需修改
# ================================================================
class MediaDisplayer:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager; self.running = True
        self.p = pyaudio.PyAudio(); self.stream = None # 延迟初始化
        print("MediaDisplayer initialized. Waiting for audio stream...")
    def _initialize_audio_stream(self):
        rate = self.data_manager.audio_rate or DEFAULT_AUDIO_RATE
        channels = self.data_manager.audio_channels or DEFAULT_AUDIO_CHANNELS
        if rate and channels:
            print(f"Creating PyAudio stream with Rate={rate}, Channels={channels}")
            self.stream = self.p.open(format=AUDIO_FORMAT, channels=channels, rate=rate, output=True, frames_per_buffer=AUDIO_CHUNK)
            return True
        return False
    def start(self):
        display_thread = threading.Thread(target=self._display_loop, daemon=True)
        display_thread.start()
    def _display_loop(self):
        print("Display loop started.")
        while self.running:
            if not self.stream and not self._initialize_audio_stream():
                time.sleep(0.1); continue
            frame = self.data_manager.latest_rgb_frame; audio_chunk = self.data_manager.latest_audio_chunk
            if frame is not None:
                cv2.imshow('接收的视频', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): self.stop(); break
            if self.stream and audio_chunk:
                self.stream.write(audio_chunk); self.data_manager.latest_audio_chunk = None
            time.sleep(0.01)
        print("Display loop stopped.")
    def stop(self):
        self.running = False; print("Stopping displayer...")
        if self.stream: self.stream.stop_stream(); self.stream.close()
        self.p.terminate(); cv2.destroyAllWindows()

# ================================================================
#  3. 媒体数据接收类 (MediaReceiver)
# ================================================================
class MediaReceiver:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.running = True
        print("MediaReceiver initialized.")

    def start(self):
        """【核心修改】启动第三个线程来接收状态数据"""
        video_thread = threading.Thread(target=self._receive_video, daemon=True)
        audio_thread = threading.Thread(target=self._receive_audio, daemon=True)
        state_thread = threading.Thread(target=self._receive_state, daemon=True) # 新增
        video_thread.start()
        audio_thread.start()
        state_thread.start() # 启动
        print("Receiver threads (video, audio, state) started.")

    def stop(self):
        self.running = False
        print("Stopping receiver...")

    def _receive_video(self):
        # (视频接收逻辑与之前完全相同)
        video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); video_socket.bind(('', VIDEO_PORT))
        print(f"视频接收端监听于 UDP port {VIDEO_PORT}..."); buffers = defaultdict(dict)
        while self.running:
            try:
                packet, _ = video_socket.recvfrom(65536)
                if len(packet) < 10: continue
                frame_id, total, idx, l = struct.unpack("!IHHH", packet[:10]); payload = packet[10:]
                if len(payload) != l: continue
                buffers[frame_id][idx] = payload
                if len(buffers[frame_id]) == total:
                    data = b"".join([buffers[frame_id][i] for i in range(total)])
                    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.data_manager.update_rgb(frame); del buffers[frame_id]
            except Exception as e: print(f"视频接收异常: {e}"); break
        video_socket.close(); print("视频接收线程已终止。")

    def _receive_audio(self):
        # (音频接收逻辑与之前完全相同)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', AUDIO_PORT)); s.listen(1)
        print(f"音频接收端监听于 TCP port {AUDIO_PORT}...")
        conn, addr = s.accept(); print(f"音频发送端已连接: {addr}")
        try:
            param_data = conn.recv(4, socket.MSG_WAITALL)
            if param_data:
                used_chans, used_rate = struct.unpack("<HH", param_data)
                self.data_manager.set_audio_params(used_rate, used_chans)
            while self.running:
                len_bytes = conn.recv(4, socket.MSG_WAITALL)
                if not len_bytes: break
                data_len = struct.unpack("!I", len_bytes)[0]
                data = conn.recv(data_len, socket.MSG_WAITALL)
                if not data: break
                self.data_manager.update_audio(data)
        except Exception as e: print(f"音频接收终止: {e}")
        finally: conn.close(); s.close(); print("音频接收线程已终止。")

    def _receive_state(self):
        """【新增方法】接收机器人状态"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', STATE_PORT))
        s.listen(1)
        print(f"状态接收端监听于 TCP port {STATE_PORT}...")
        
        conn, addr = s.accept()
        print(f"状态发送端 (机器人) 已连接: {addr}")
        
        try:
            while self.running:
                # 1. 接收4字节的长度前缀
                len_bytes = conn.recv(4, socket.MSG_WAITALL)
                if not len_bytes:
                    break # 连接关闭
                
                payload_len = struct.unpack("!I", len_bytes)[0]
                
                # 忽略心跳包
                if payload_len == 0:
                    continue
                
                # 2. 接收确切长度的数据负载
                payload = conn.recv(payload_len, socket.MSG_WAITALL)
                if not payload:
                    break
                
                # 3. 解包数据
                x, y, yaw = struct.unpack('!ddd', payload)
                
                # 4. 更新 DataManager
                state = np.array([x, y, yaw], dtype=np.float32)
                self.data_manager.update_state(state)

        except Exception as e:
            print(f"状态接收终止: {e}")
        finally:
            conn.close()
            s.close()
            print("状态接收线程已终止。")


# ================================================================
#  主程序入口 (Main Execution)
# ================================================================
if __name__ == "__main__":
    data_manager = DataManager()
    receiver = MediaReceiver(data_manager)
    receiver.start()
    displayer = MediaDisplayer(data_manager)
    displayer.start()

    def model_inference_loop():
        while displayer.running:
            time.sleep(2)
            if data_manager.audio_rate:
                print("\n--- 模拟大模型调用 ---")
                model_input = data_manager.get_model_input()
                
                # 打印所有模态的信息
                rgb_shape = model_input["rgb"].shape
                audio_shape = model_input["audio"].shape
                state_val = model_input["state"]
                
                print(f"成功获取模型输入:")
                print(f"  RGB shape: {rgb_shape}, dtype: {model_input['rgb'].dtype}")
                print(f"  Audio shape: {audio_shape}, dtype: {model_input['audio'].dtype}")
                print(f"  State value: [\nx={state_val[0]:.16f}, \ny={state_val[1]:.16f}, \nyaw={state_val[2]:.16f}\n], dtype: {state_val.dtype}")
                print("----------------------\n")
    
    model_thread = threading.Thread(target=model_inference_loop, daemon=True)
    model_thread.start()

    try:
        while displayer.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n接收到Ctrl+C，程序退出。")
    finally:
        displayer.stop()
        receiver.stop()
        print("所有服务已关闭。")