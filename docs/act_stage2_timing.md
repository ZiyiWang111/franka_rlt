# Stage 2 耗时日志

正常启动 `act_rlt/train_stage2.py` 即自动启用，逐 chunk 写入运行输出目录的
`metrics.jsonl`，无需新增命令行参数。建议保留 `config.json` 和启动命令，
并采集 warmup、正式训练各一段，以比较训练竞争的影响。

所有 `*_ms` 单位均为毫秒。`*_wall_ms` 是主机墙钟时间；GPU 运算异步提交，
不能将主机提交时间当成 GPU 执行时间。`*_cuda_ms` 使用当前 CUDA stream 上的
事件计时，包含两事件间的设备等待/竞争，不是独占 GPU 的纯计算耗时。
读取事件不会强制同步；事件尚未完成时写 `null`，CPU 路径不生成 CUDA 字段。
计时有少量调用开销，未额外插入 CUDA synchronize。

## 当前 chunk

- `current_act_{wall,cuda}_ms`、`current_rl_token_{wall,cuda}_ms`：生成本段动作时的 ACT、RL Token 耗时。
- `current_encode_{wall,cuda}_ms`：ACT、RL Token、状态拼接的整体耗时，包含上述子项，不应相加。
- `current_actor_load_{wall,cuda}_ms`：检查并加载参数快照；没有新快照时仅为检查开销。
- `current_actor_{wall,cuda}_ms`：actor 推理；warmup/人工控制时不生成此项。
- `current_postprocess_cpu_wall_ms`：三组动作反归一化及转 CPU，包含这里等待 GPU 完成的时间。
- `current_command_prepare_wall_ms`：动作限幅、归一化执行记录和命令字典构建。
- `current_prediction_wall_ms`：actor 参数加载到命令准备的总耗时，包含上述子项。
- `prediction_queue_wait_wall_ms`：servo 等预测队列的时间。
- `prediction_ready_to_first_send_ms`：预测准备好到本段第一条指令发送返回。
- `chunk_boundary_command_interval_ms`：上一段最后一条与本段第一条发送返回的间隔；首段没有此项。
- `control_period_ms`：计划控制周期。边界指令间隔超出此值的部分可用于估算额外停顿。
- `max_command_interval_ms`：本段观察到的最大指令间隔，包含进入本段的边界。
- `send_action_wall_ms_sum/max`：本段发送调用总耗时/最大耗时；不包含 resync。
- `resync_wall_ms_sum`：本段发送前重同步命令位姿的总耗时。
- `boundary_finalize_wall_ms`：本段末尾处理时间，包括按需等最后一拍、停止 servo、控制权交接及结果构建。

## 当前 chunk 结束后，准备下一段

- `next_request_queue_wall_ms`：边界请求到推理线程开始处理。
- `next_camera_wait_wall_ms`：等待符合边界新鲜度要求的观测。
- `next_camera_capture_wall_ms`：被选中观测的 get_observation 调用时间；是上述等待的相关诊断项，不能直接累加。
- `next_observation_start_after_request_ms`：被选中观测开始采集时间相对边界请求的偏移。这里是主机调用时间，不是相机硬件曝光时间。
- `next_preprocess_wall_ms`：观测组帧和预处理的主机耗时。
- `next_act_*`、`next_rl_token_*`、`next_encode_*`、`next_actor_*`、`next_postprocess_cpu_wall_ms` 等：下一段的推理准备，与下一行的 `current_*` 对应。
- `next_boundary_total_wall_ms`：边界请求到下一段命令准备并入队的总耗时。最后一段仅编码终态，不生成下一段动作，因此不能与普通边界直接比较。
- 原有 `boundary_inference_ms` 保留，计时截止于 encode 返回，不包含之后的 actor 和命令准备。

`next_boundary_total_wall_ms` 不包含 `boundary_finalize_wall_ms`，也不等同于实际指令间隔。
实际停顿应查看下一行 `chunk_boundary_command_interval_ms`。

## Learner

- `actor_publish_wall_ms`：本次 learner 调用发布 actor CPU 快照的耗时；结果队列有积压时，不表示该结果所用 actor 的发布时间。
- `learner_updates_wall_ms`：本 chunk 对应的整组 UTD 更新墙钟耗时；warmup 或 replay 未满时无此项。
- `result_queue_wait_wall_ms`、`result_queue_depth`：结果等待 learner 消费的时间和消费后的队列长度，可识别 learner 落后。
- `warmup`、`control_source`、`collector_paused`、`done`：分析时分组，排除人工接管和阶段结束边界。

## 相机边界优化（默认开启）

`train_stage2.py` 默认使用 `--camera-boundary-mode timestamp`，原训练命令无需
增加参数。A/B 对比可指定 `--camera-boundary-mode legacy`。无需重启控制服务器。

采集线程持续读取相机，保留最近 8 份观测。边界选择缓冲区中第一份满足新鲜度
条件的观测；即使采集调用开始于边界之前，只要可验证的曝光开始时间和配套
机器人状态时间都在边界之后，也可采用。该观测编码仍同时作为 replay 的
`next_state` 和下一段 actor 输入，未添加预测动作、重放动作或修改训练更新规则。

时间处理遵循 [RealSense SDK 时间域及元数据定义](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h)：

- 启动时尝试开启 SDK global time；只接受 `global_time`，不把 `system_time`
  （可能是到达时间）或未转换的 `hardware_clock` 当作可靠曝光时间。
- 通过 frame_timestamp、sensor_timestamp 和 actual_exposure 元数据还原曝光
  中点与开始时间，将 SDK 全局时间转换为主机 monotonic 时间。曝光开始时间
  再减 2 ms 裕量用于边界判断。此裕量是工程保护值，SDK 时间映射不是硬件同步
  精度保证；实际设备表现需检查日志。
- 连续 3 帧通过检查后采用新路径；拒绝重复/倒退帧、异常帧龄、非法曝光数据、
  主机时钟跳变、设备到全局时钟映射跳变。元数据不完整则自动回退。
- 控制客户端保留最近 256 份 PUSH 状态，选取距图像曝光中点最近且相差不超过
  20 ms 的有效快照。只在控制服务器地址为 localhost/127.0.0.1/::1 时使用服务器
  monotonic 时间戳；远程服务器未做时钟同步，自动回退。
- 状态时间是服务器快照时间，不是机器人硬件采样时间；记录对齐偏差，不宣称
  实现精确硬件同步。双相机曝光中点相差超过 33.3 ms 也回退。
- 回退保留原来在读图前读取的机器人状态，按观测调用开始时间筛选，原因写入
  JSONL。相机读取超时、控制异常仍沿用原有异常和停止流程。

新增字段（同样有 `current_` / `next_` 前缀）：

| 字段 | 含义 |
|---|---|
| `camera_boundary_mode` | `exposure_timestamp`：优化生效；`legacy_fallback`：本次自动回退；`legacy`：显式关闭或接口不支持 |
| `camera_boundary_fallback_reason` | 自动回退原因；正常优化时为 null |
| `observation_fresh_after_request_ms` | 曝光开始时间（减裕量）及状态时间的较早者，相对边界请求的偏移 |
| `camera_state_skew_ms` | 状态快照时间减图像曝光中点；正数表示状态较晚 |
| `camera_frame_age_ms` | 配对时曝光中点的帧龄 |
| `camera_frame_spread_ms` | 多相机曝光中点跨度；单相机为 0 |
| `camera_timestamp_margin_ms` | 帧时间判断的保护裕量 |

优化生效时，`observation_start_after_request_ms` 可以是负数，这表示采集调用
早已开始；`observation_fresh_after_request_ms` 应非负，表示实际采用的图像和
状态满足边界条件。比较速度时按 `camera_boundary_mode` 分组，排除首段、
终止段和人工切换；不要把回退样本当作优化已生效。
