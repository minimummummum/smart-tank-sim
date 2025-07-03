import math
import numpy as np
import random

class Aim():
    def __init__(self):
        pass
    def reset(self):
        pass
    def get_action(self, log_data):
        self.tank_x = log_data.get("playerPos", {}).get("x")
        self.tank_y = log_data.get("playerPos", {}).get("y")
        self.tank_z = log_data.get("playerPos", {}).get("z")
        self.enemy_x = log_data.get("enemyPos", {}).get("x")
        self.enemy_y = log_data.get("enemyPos", {}).get("y")
        self.enemy_z = log_data.get("enemyPos", {}).get("z")
        # 테스트
        self.enemy_x = 234.61102294921876
        self.enemy_y = 6.604026794433594
        self.enemy_z = 229.89015197753907
        self.speed = log_data.get("playerSpeed")
        # 포탑 각도
        self.turret_x = log_data.get("playerTurretX")
        self.turret_y = log_data.get("playerTurretY")
        return self._get_action()
    def angle_diff(self, a, b):
        """두 각도 사이의 최소 차이 (0~180도)"""
        return (a - b + 180) % 360 - 180

    def find_pitch_for_target_distance(self, x, y, z, yaw_deg, target_x, target_y, target_z, gravity=9.81, tolerance=1e-5, max_iterations=50):
        initial_speed = 54
        turret_length = 5.891
        turret_offset = turret_length / 2
        # y 좌표는 포탑 높이를 고려하여 조정
        #y -= 5
        y -= target_y - 3
        # yaw를 라디안으로 변환
        yaw = math.radians(yaw_deg)

        # yaw 방향으로의 수평 거리 계산
        dx = target_x - (x + turret_offset * math.sin(yaw))
        dz = target_z - (z + turret_offset * math.cos(yaw))
        horizontal_distance = math.sqrt(dx**2 + dz**2)

        # 목표 거리가 0이거나 음수인 경우
        if horizontal_distance <= 0:
            return None

        def get_distance_for_pitch(pitch_deg):
            pitch = math.radians(pitch_deg)
            # 포탑 높이 변화 반영
            adjusted_y = y + turret_offset * math.sin(pitch)
            # 방향 벡터 계산
            vy = initial_speed * math.sin(pitch)
            vxz = initial_speed * math.cos(pitch)
            # 착탄 시간 계산
            a = -0.5 * gravity
            b = vy
            c = adjusted_y
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return float('inf')
            t_impact = (-b + math.sqrt(discriminant)) / (2 * a)
            if t_impact < 0:
                t_impact = (-b - math.sqrt(discriminant)) / (2*a)
                if t_impact < 0:
                    return float('inf')
            # 착탄 수평 거리
            return vxz * t_impact

        def objective(pitch_deg):
            return get_distance_for_pitch(pitch_deg) - horizontal_distance

        # Newton-Raphson
        pitch_deg = 10  # 초기 추정값
        for _ in range(max_iterations):
            f = objective(pitch_deg)
            if abs(f) < tolerance:
                return pitch_deg
            # 수치적 도함수
            delta = 0.001
            f_prime = (objective(pitch_deg + delta) - objective(pitch_deg - delta)) / (2 * delta)
            if abs(f_prime) < 1e-10:  # 도함수가 0에 가까우면 중단
                return None
            pitch_deg -= f / f_prime
            if not -20 <= pitch_deg <= 45:  # 범위 제한
                return None
        
        return None
    def _get_action(self):
        """스크립트된 행동: 포탑을 적 방향으로 조준하고 발사"""
        dx = self.enemy_x - self.tank_x
        dz = self.enemy_z - self.tank_z
        distance = math.sqrt(dx**2 + dz**2)
        target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        target_pitch = self.find_pitch_for_target_distance(self.tank_x, self.tank_y, self.tank_z, self.turret_x, self.enemy_x, self.enemy_y, self.enemy_z)
        if target_pitch:
            yaw_error = self.angle_diff(self.turret_x, target_yaw)
            pitch_error = target_pitch - self.turret_y
            speed = self.speed
            if yaw_error > 30:
                turret_dx = 2.0
            elif yaw_error < -30:
                turret_dx = -2.0
            elif yaw_error > 10:
                turret_dx = 1
            elif yaw_error < -10:
                turret_dx = -1 
            elif yaw_error > 0:
                turret_dx = yaw_error * 0.2
            elif yaw_error < 0:
                turret_dx = yaw_error * 0.2
            else:
                turret_dx = 0.0
            
            if pitch_error > 3:
                turret_dy = 1.0
            elif pitch_error < -3:
                turret_dy = -1.0
            elif pitch_error > 0:
                turret_dy = pitch_error * 0.5
            elif pitch_error < -0:
                turret_dy = pitch_error * 0.5
            else:
                turret_dy = 0.0
            turret_dx *= 1 - distance/1500
            #turret_dy = 0.3  if pitch_error > 0 else (-0.3 if pitch_error < -0 else 0)
            fire = 1 if abs(yaw_error) < 1 and abs(pitch_error) < 1 else 0
            action = [turret_dx, turret_dy, fire]

            return action
        else:
            return None
