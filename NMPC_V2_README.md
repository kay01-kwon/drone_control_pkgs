# NMPC Node V2 - ROS2 완전 재작성

## 개요

기존 `nmpc_only_node`가 ROS1과 다른 결과를 보여주는 문제를 해결하기 위해 완전히 새로 작성한 ROS2 NMPC 노드입니다.

## 주요 변경사항

### 1. ROS2 Native 구조
- **Threading 제거**: 기존의 `threading.Thread` 대신 ROS2의 `create_timer` 사용
- **SingleThreadedExecutor**: 예측 가능한 동작을 위해 단일 스레드 실행
- **Timer 기반 제어 루프**: ROS2 스타일의 100Hz timer callback

### 2. 깔끔한 State 관리
- Lock 없이도 안전한 state 접근
- 명확한 초기화 순서
- Odometry 데이터 freshness 체크

### 3. 향상된 로깅 및 통계
- 평균 solver 시간 추적
- Success rate 모니터링
- Odometry 지연 경고
- 주기적인 통계 출력 (1초마다)

### 4. Robust Error Handling
- Solver 실패 처리
- Stale data 감지
- Graceful shutdown

## 파일 구조

```
drone_control/
├── drone_control/
│   ├── nmpc_only_node.py      # 기존 노드 (ROS1 스타일)
│   └── nmpc_node_v2.py         # 새 노드 (ROS2 Native)
├── launch/
│   ├── nmpc_only.launch.py    # 기존 launch
│   └── nmpc_v2.launch.py      # 새 launch
└── config/
    └── nmpc_only.yaml         # 공통 설정 파일
```

## 사용법

### 빌드

```bash
cd ~/your_workspace
colcon build --packages-select drone_control --symlink-install
source install/setup.bash
```

### 실행

```bash
# 새 V2 노드 실행
ros2 launch drone_control nmpc_v2.launch.py

# 또는 직접 실행
ros2 run drone_control nmpc_node_v2 --ros-args --params-file path/to/nmpc_only.yaml
```

### 기존 노드와 비교

```bash
# 기존 노드
ros2 launch drone_control nmpc_only.launch.py

# 새 V2 노드
ros2 launch drone_control nmpc_v2.launch.py
```

## 주요 차이점: ROS1 스타일 vs ROS2 Native

| 항목 | 기존 (nmpc_only_node) | 새로운 (nmpc_node_v2) |
|------|----------------------|----------------------|
| 제어 루프 | threading.Thread | create_timer |
| Executor | MultiThreadedExecutor | SingleThreadedExecutor |
| Rate control | time.sleep() | ROS2 Timer |
| Thread safety | Manual locking 필요 | Lock-free (single thread) |
| 예측 가능성 | 낮음 (thread scheduling) | 높음 (timer callback) |

## 성능 모니터링

노드는 1초마다 다음 통계를 출력합니다:

```
Stats: solve=2.35ms, success=100.0%, odom_age=5.2ms
```

- `solve`: 평균 solver 실행 시간 (ms)
- `success`: Solver 성공률 (%)
- `odom_age`: Odometry 데이터 나이 (ms)

## 문제 해결

### Solver가 자주 실패하는 경우

1. Horizon이나 weight 조정:
   ```yaml
   nmpc_param:
     t_horizon: 1.0  # 더 짧게 시도
     QArray: [...]   # Weight 조정
   ```

2. Odometry 품질 확인:
   - `odom_age`가 50ms 이상이면 경고 출력
   - `/filtered_odom` 토픽의 발행 주기 확인

### ROS1과 결과가 다른 경우

새 V2 노드는 다음을 보장합니다:

1. **일정한 제어 주기**: Timer가 정확히 100Hz 유지
2. **예측 가능한 실행 순서**: Single thread로 callback 순서 보장
3. **동일한 solver 설정**: ROS1과 동일한 Acados 파라미터 사용

## 기술적 세부사항

### Velocity Frame 변환

Odometry는 body frame velocity를 제공하지만, NMPC는 world frame이 필요합니다:

```python
# Body -> World frame 변환
v_body = state_current[3:6]
q = state_current[6:10]
R_world_body = quaternion_to_rotm(q)
v_world = R_world_body @ v_body
```

### Reference Quaternion

Yaw 각도에서 quaternion 변환:

```python
# Yaw only rotation
qw = cos(psi/2)
qx = 0
qy = 0
qz = sin(psi/2)
```

### Thrust to RPM

```python
# T = C_T * omega^2
# omega = sqrt(T / C_T)
rpm = sqrt(thrust / C_T)
```

## 향후 개선 가능 항목

1. **Adaptive rate**: Solver 시간에 따라 제어 주기 조정
2. **Trajectory preview**: 미래 reference trajectory 고려
3. **State estimation integration**: EKF/UKF와의 tight coupling
4. **Dynamic reconfigure**: 실행 중 파라미터 변경

## 라이선스

원본 프로젝트 라이선스 따름

## 작성자

- Claude (AI Assistant)
- 날짜: 2025-12-28
