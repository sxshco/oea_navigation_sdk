# Go2 Navigation MCP

这是一个可注册的 MCP 服务器，用于把文档中的目标搜索、目标重定位、局部避障和导航状态管理封装成 agent 可调用工具。

## 已实现能力

- 基于 RGB 观测进行目标检测，并在有深度时优先使用深度、无深度时回退到单目距离估计
- 当前视野未发现目标时，按 45 度步进顺时针搜索，最多一整圈
- 使用 2D 局部占据图进行逐步重规划避障，并结合机器人端连续闭环速度控制做本地避障
- 进入成功距离阈值且机身朝向目标后判定为成功
- 被障碍物阻挡时，会先逼近最近可行位置；只有目标附近完全无可行接近区域时才返回阻挡
- 检测到目标位置大幅变化时，自动重新追踪并规划
- 以标准输入输出方式运行的 MCP JSON-RPC 服务，可被 agent 注册
- 主机端可通过 SSH 自动同步机器人端脚本、完成交互式登录并远程拉起服务

## 目录

- `navigation_mcp/server.py`: MCP `stdio` 服务入口
- `navigation_mcp/navigator.py`: 导航状态机与搜索/重规划逻辑
- `navigation_mcp/perception.py`: 目标检测、深度反投影和无深度单目距离回退
- `navigation_mcp/bridge.py`: 主机端桥接，接收 `data_server.py` 的视频/深度/状态/占据图并向 `atom_server.py` 发送控制
- `move/data_server.py`: 机器人端数据服务，发送 RGB、深度、状态和占据图
- `move/atom_server.py`: 机器人端控制服务，回连主机端 websocket 控制器，并通过官方 `unitree_sdk2_python` 下发连续速度控制

## 快速启动

```bash
cd /home/ly/songxinshuai/navigation_sdk
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd sam3_main
pip install -e .
cd ..
pip install -r requirements.txt
python -m navigation_mcp.server
```

或安装后：

```bash
cd /home/ly/songxinshuai/navigation_sdk
pip install -e .
go2-navigation-mcp
```

## 建议的 MCP 注册命令

```json
{
  "command": "python",
  "args": ["-m", "navigation_mcp.server"],
  "cwd": "/home/ly/songxinshuai/navigation_sdk"
}
```

## 可用工具

- `set_navigation_target`: 设置目标、成功距离/朝向阈值与可选检测提示
- `configure_go2_backend`: 配置主机监听端口、SSH 目标、机器人端工作目录和动作后端模式
- `start_go2_services`: 通过 SSH 在机器狗端启动 `data_server.py` 和 `atom_server.py`
- `stop_go2_services`: 停止 SSH 拉起的机器人端通信/控制进程
- `step_navigation`: 执行一步观测、检测、规划和动作决策
- `run_navigation`: 循环执行直到成功、失败或超时
- `cancel_navigation`: 取消当前任务
- `get_navigation_status`: 查看当前导航状态
- `load_observation`: 手动注入一帧 RGB/Depth/Occupancy 观测，便于离线调试

## 真实机器人对接说明

真实部署时，主机端运行 MCP 和 SDK Python 环境，机器人端运行两个进程：

1. `move/data_server.py`：向主机发送 RGB 视频、对齐深度、相对位姿和雷达占据图；相机采集优先走 RealSense，失败后回退到 OpenCV。
2. `move/atom_server.py`：连接主机端 websocket 控制器，并在两种动作后端之间切换：
   - `sdk2`：使用官方 `unitree_sdk2_python` 的 `SportClient`
   - `ros2_topic`：向 `/api/sport/request` 发布 `unitree_api/msg/Request`
   同时继续基于里程计与激光点云执行闭环控制。

当前实现会把 RealSense 对齐深度通过独立 TCP 通道回传给主机；如果深度通道暂时不可用，导航器仍会自动回退到基于目标框面积的单目距离估计。

当前局部前进命令的最小步长默认是 `0.10m`，主机端动作确认的最小平移阈值默认是 `0.05m`。这样近距离前进时不会因为只发出 `0.05m` 小步，却要求 `0.08m` 位移确认而被误判为“未运动”。

当前避障是双层的：

1. 主机端 `navigation_mcp/navigator.py` 每个导航 step 都会读取最新占据图，重新规划到目标成功半径附近或最近可达位置，因此对新出现障碍具有持续响应能力。
2. 机器人端 `move/atom_server.py` 在执行闭环平移动作时，会基于最新 mid360 点云判断前方、左侧和右侧是否可通行，必要时做本地侧移绕障；真正的运动指令可由官方 SDK 下发，也可走 ROS2 topic 下发。

因此它已经具备基于最新传感器数据的准实时避障能力。当前实现不是旧版键盘控制那种定时原子动作，而是“主机下发目标距离/目标转角，机器人端通过官方 SDK 做连续速度控制直至完成”的模式。

## 当前 Go2 启动模型

当前项目与旧版手工控制流程相比，有这几个关键差异：

1. 主机端不再需要单独手工常驻 `move/atom_client.py` 和 `move/data_client.py`。
2. 主机端 `navigation_mcp/bridge.py` 会直接承担：
   - websocket 动作控制器
   - RGB / 状态 / 占据图接收器
   - SSH 远程同步与拉起器
3. `start_go2_services` 调用后，主机会：
   - 通过 `scp` 将本地代码同步到机器狗端指定目录
   - 通过交互式 `ssh` 自动输入密码
   - 自动选择 `ros:foxy`
   - 自动输入 `sudo` 密码
   - 先启动 MID360 对应环境和驱动
   - 再启动机器人端 `data_server.py` 与 `atom_server.py`
   - 启动后额外检查远端相关进程，以及 `/api/sport/request` 的 topic 信息

## 重要前提

当前机器狗登录链路不是纯净的非交互式 shell，而是会经历：

1. SSH 密码输入
2. `ros:foxy(1) noetic(2) ?` 选择
3. `sudo` 密码输入

当前项目已经在主机侧桥接层里兼容了这条交互链路，因此不需要修改机器狗上的 `.bashrc`。但这也意味着，如果登录提示文案变化，自动化登录逻辑可能需要同步调整。

另外，当前机器狗环境下还必须显式 `source` 两套环境：

1. 启动雷达前：
   `source /home/unitree/mid-360/ws_livox/install/setup.sh`
2. 启动控制与数据节点前：
   `source /home/unitree/unitree/sunqianran/SemGo2/rpc.sh`

这两步都必须在远程启动参数中提供，否则机器人端对应节点很可能起不来。

第 2 条是当前更推荐的值。`rpc.sh` 不仅会 source ROS2 workspace，还会额外加载 `cyclonedds_ws`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、`CYCLONEDDS_URI`、`Go2_ROS2_example` 和 RealSense Python 路径；当前实机排查表明，之前直接 source `unitree_ws/install/setup.bash` 时出现的 `bad_alloc`，改为这条后可以规避。

另外，若使用 `rpc.sh`，建议让 `livox` 启动命令也先进入同一套环境。否则 `livox/lio-sam` 发布者与 `data_server.py` 订阅者可能落在不同的 DDS/RMW 环境里，导致 `/livox/lidar` 和 `/utlidar/robot_odom` 在机器人端看似存在、但 `data_server.py` 实际订阅不到。

建议流程：

1. 在主机启动 MCP 服务。
2. 调用 `configure_go2_backend`，填入主机 IP、机器人 SSH 地址、密码、ROS 选择、远程项目目录和 `source` 命令。
3. 调用 `start_go2_services`。
4. 先确认：
   - `motion_connected` 已变为 `true`
   - `latest_observation_timestamp` 在更新
5. 再调用 `set_navigation_target` 和 `run_navigation`。

## 推荐调用顺序

1. 启动 MCP：

```bash
cd /home/ly/songxinshuai/navigation_sdk
source .venv/bin/activate
python -m navigation_mcp.server
```

2. 在 agent 中调用 `configure_go2_backend`
3. 调用 `start_go2_services`
4. 调用 `set_navigation_target`
5. 调用 `run_navigation`

所有真实机器人调用都建议显式传 `"backend": "go2"`，不要依赖默认后端。

一个典型的 `set_navigation_target` 参数例子：

```json
{
  "backend": "go2",
  "target_label": "cup",
  "success_distance_m": 0.6,
  "success_heading_deg": 8.0,
  "detection_hint": {
    "rgb_range": [[180, 0, 0], [255, 60, 60]],
    "min_pixels": 150
  }
}
```

## 适配当前机器狗的 `configure_go2_backend` 示例

下面这份配置对应当前已确认的环境：

- 主机局域网 IP：`192.168.86.106`
- 机器狗 IP：`192.168.86.26`
- SSH 用户：`unitree`
- SSH 密码：`123`
- sudo 密码：`123`
- ROS 选择：`foxy`，即 `1`

```json
{
  "host_bind": "192.168.86.106",
  "video_port": 5220,
  "state_port": 5222,
  "occupancy_port": 5223,
  "depth_port": 5224,
  "motion_port": 8000,

  "ssh_host": "192.168.86.26",
  "ssh_user": "unitree",
  "ssh_password": "123",
  "remote_ros_choice": "1",
  "remote_sudo_password": "123",

  "remote_project_dir": "/home/unitree/navigation_sdk",
  "remote_python": "python3",
  "remote_sync_before_start": true,

  "remote_livox_setup": "source /home/unitree/mid-360/ws_livox/install/setup.sh",
  "remote_livox_launch": "ros2 launch livox_ros_driver2 rviz_MID360_launch.py",

  "remote_setup": "source /home/unitree/unitree/sunqianran/SemGo2/rpc.sh",

  "remote_data_script": "move/data_server.py",
  "remote_data_video_backend": "realsense",
  "remote_data_video_index": 4,
  "remote_motion_script": "move/atom_server.py",
  "remote_motion_backend": "auto",
  "remote_motion_sdk_python_path": "/home/unitree/unitree_sdk2_python",
  "remote_motion_network_interface": "wlan0",
  "remote_motion_require_subscriber": true,

  "remote_startup_delay_s": 4.0,
  "remote_observation_wait_timeout_s": 20.0
}
```

## 如果需要改为运行机器狗端已有 ROS2 包

如果你最终确认机器人端必须运行已有的 `http_control` 包，而不是同步过去的 Python 脚本，可以改用：

```json
{
  "remote_data_command": "source /home/unitree/unitree/sunqianran/install/setup.sh && ros2 run http_control data_server",
  "remote_motion_command": "source /home/unitree/unitree/sunqianran/install/setup.sh && ros2 run http_control atom_server"
}
```

这时请注意：

1. `remote_*_command` 与 `remote_*_script` 是两种不同模式。
2. 如果走 `ros2 run http_control ...`，请确保机器狗端该 ROS2 包内容已经是你需要的版本。

## 动作后端与诊断

`move/atom_server.py` 现在支持三种模式：

- `auto`：先尝试 `sdk2`，失败后回退到 `ros2_topic`
- `sdk2`：强制使用官方 SDK
- `ros2_topic`：强制发布到 `/api/sport/request`

如果你怀疑之前 ROS2 方案失败，不要只看“atom_server/data_server 进程是否存在”。更关键的是：

- `data_server` 只负责视频、位姿和占据图上行，它正常不代表底层运动链路正常
- `atom_server` 就算成功启动，也可能只是把 `Request` 发布到了一个没有订阅者的 `/api/sport/request`
- 这种情况下，主机端过去会看到“命令已发出”，但里程计始终不变；现在会在机器人端直接拒绝该命令，并把 `no_subscriber_on_/api/sport/request` 通过 websocket 回传

## 相机链路

`move/data_server.py` 现在支持三种视频输入模式：

- `realsense`：优先使用 RealSense 对齐彩色流获取彩色帧
- `opencv`：回退到原来的 `cv2.VideoCapture`
- `auto`：先试 `realsense`，失败再回退 `opencv`

如果当前 D435i USB 链路稳定，建议把 `remote_data_video_backend` 设为 `realsense`，通常会比纯 `VideoCapture` 更稳定、预热更短、感知延迟更低。

另外，真实机器人上观测链路往往比动作 websocket 更慢 ready。`start_go2_services` 现在会额外等待初始观测到达，等待上限由 `remote_observation_wait_timeout_s` 控制；如果你过早检查状态，可能会误以为“观测未返回”，但实际只是 RealSense、里程计或占据图上行仍在初始化。

## 搜索旋转策略

当前默认搜索策略会在每次未检测到目标时顺时针旋转 `45` 度，再做下一次检测；整圈共 `8` 次离散搜索。这样比之前每次 `90` 度、共 `4` 次的策略更不容易因为角度采样过粗或实机转向误差而漏检。

## 当前机器人端 SDK 依赖

当前控制脚本默认会在机器狗上使用：

- `/home/unitree/unitree_sdk2_python`

这是当前已实际确认存在的官方 SDK Python 目录；如果你的机器狗目录不同，可通过 `configure_go2_backend` 里的 `remote_motion_sdk_python_path` 覆盖。若 SDK 必须绑定特定网卡，可再传 `remote_motion_network_interface`，当前机器狗上已确认可用的无线网卡名为 `wlan0`。

注意：在当前这台机器狗上，直接运行系统 `python3` 导入 `unitree_sdk2py` 仍会报 `ModuleNotFoundError: No module named 'cyclonedds'`。也就是说，SDK 代码路径已经具备，但默认 Python 运行环境还需要补齐 `cyclonedds` Python 依赖，或者切换到一个已经带该依赖的 Python 环境后再启动 `move/atom_server.py`。

## 已知隐患

1. `start_go2_services` 当前能确认“远端命令已进入启动阶段”，但不等于每个 ROS 节点都已完整 ready。
2. SAM3 依赖路径仍有硬编码；如果新主机没有对应环境，请为目标设置 `detection_hint`，或先修正 `navigation_mcp/perception.py` 中的模型环境路径。
3. 默认 MCP 后端仍然是 `sim`，真实机器人调用务必显式传 `"backend": "go2"`。
4. 视频设备索引默认是 `4`；如果 D435i 在机器狗上枚举顺序变化，需要调整 `move/data_server.py` 的 `--video-index` 或默认值。
