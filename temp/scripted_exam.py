def scripted_patrol_action(state):
    # state에서 엄폐물 위치, 적 위치 정보 확인
    # 엄폐물 쪽으로 이동 명령 (W,A,S,D 중 하나)
    # 포탑 움직임 최소화 (Q,E,R,F 모두 0)
    # 포탄 발사 안 함 (spacebar=0)
    action = {
        'move_forward': 0.8,
        'turn_left': 0.2,
        'turret_left': 0.0,
        'fire': False,
        # 기타 필요한 행동 조절
    }
    return action

if current_command == 'patrol' and random.random() < 0.2:
    action = scripted_patrol_action(state)
else:
    action = ppo_policy(state)