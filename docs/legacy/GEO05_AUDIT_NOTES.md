# GEO-0.5 审计说明

## 计数

`data/sweeps_geo05` 有 960 个 RGB-D 帧。每帧包含 LEFT、RIGHT、TOP、BOTTOM 四个独立边界标签，因此按边界实例统计时总数可以超过帧数；`geo05_validation.json` 同时保存了按唯一帧去重后的状态计数。

## RIGHT 轨迹

RIGHT 使用与 LEFT 相同的 NORMAL_LOCK 规则，方向只改变水平位置，姿态保持固定。`facade_a` 在 5m 轨迹出现完整 `IN -> STRADDLE -> OUT`；10m/15m 因立面宽度与视场关系从 `STRADDLE -> OUT` 开始。`facade_d` 的 47m 宽实体立面在这些距离主要保持 IN，随后离开视场进入 OUT，因此没有稳定的 STRADDLE 段。`trajectory_transition_audit.json` 记录了 12 条轨迹的完整压缩状态序列。

## 标签语义

一帧可以同时有四个边界标签。`IN` 要求 `visible_ratio >= 0.95` 且该物理边界不在图像内部；`STRADDLE` 要求物理边界在图像内部、目标侧和外部侧探针都可见；`OUT` 要求 `visible_ratio <= 0.05`；其余为 `UNKNOWN`。

`UNKNOWN` 覆盖投影退化、采样点无效、遮挡、边界在图像内但两侧探针不能同时确认等情况。

## 深度与可见率

遮挡判断使用 `abs(sensor_depth - theoretical_camera_z) <= max(0.15m, 0.02 * theoretical_camera_z)`。`visible_ratio` 的分母是投影落入当前图像的立面采样点数；视锥外采样点不计为遮挡，而用于判断物理边界是否在视野外。最终深度语义采用 z-depth，独立平面验证结果见 `results/geo05r2/depth_metric_v2.json`。

## 测试

`tests/` 覆盖几何、标签和验证门控；当前使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest -q tests`，不联网安装插件。
