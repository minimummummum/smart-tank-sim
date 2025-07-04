from flask import Flask, request, jsonify, render_template
from multiprocessing import Process, Queue
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import logging
import queue
from core.utils import clear_queue
import time
import math
import google.generativeai as genai
import core.path_Finder as pf 
import core.aim as aim
import core.ppo as ppo
from collections import Counter, deque
app = Flask(__name__)
CORS(app)
lidar_data = []
obstacles = []
player_pos = []
player_lidar_angle_z = []
tank_status_data = {
    "x": 0,
    "z": 0,
    "yaw": 0,
    "pitch": 0,
    "impact_x": 0,
    "impact_z": 0,
    "real_impact_x": 0,
    "real_impact_z": 0,
    "goal_x":None,
    "goal_z":None,
    "bot_command":None,
    "user_command":None,
    "path":[]
}
obstacle_data = {}
chat_history = []  # 채팅 기록 저장 리스트

yolo_input_queue = Queue(maxsize=1)
yolo_output_queue = Queue(maxsize=1)
action_input_queue = Queue(maxsize=1)
action_output_queue = Queue(maxsize=1)
detect_input_queue = Queue(maxsize=1)
init_input_queue = Queue(maxsize=1)
hit_input_queue = Queue(maxsize=1)
collision_input_queue = Queue(maxsize=1)
info_input_queue = Queue(maxsize=1)
info_output_queue = Queue(maxsize=1)
obstacles_input_queue = Queue(maxsize=1)
rl_output_queue = Queue()
target_point_queue = Queue(maxsize=1)

# target_classes = {0: "Car", 3: "E_Tank", 4: "Human"}
target_classes = {1: "Car", 0: "E_Tank", 2: "Human"}
def yolo_worker(yolo_input_q, yolo_output_q):
    # model = YOLO("yolov8x_e500_s512_b8.pt").to("cuda")
    model = YOLO("yolom_e1000_i640_b8_es100.pt").to("cuda")
    # YOLO 프로세스 반복
    while True:
        # /detect request yolo_input_q에서 이미지 가져오기
        image = yolo_input_q.get()
        results = model(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy().tolist()
        # YOLO 결과를 yolo_output_q에 넣어 /detect로 response
        yolo_output_q.put(detections)


def action_worker(action_input_q, action_output_q, hit_input_q, detect_input_q,
                  info_input_q, info_output_q, init_input_q, collision_input_q, 
                  obstacles_input_q, rl_output_q, target_point_q):
    hit_data = None
    detections = None
    init_data = None
    collision_data = None
    astar = pf.Path()
    aim_bot = aim.Aim()
    rl = ppo.Agent()
    target_point = None
    supply_point = (203, 56)
    base_point = (8, 290)
    enemy_point = (235.81, 6.60, 290.74)
    enemys_point = (146, 160)
    cars_point = (252, 265)
    # supply_point = (290, 286)
    # base_point = (30, 295)
    # enemy_point = (236.4, 8.2, 157.7)
    # enemys_point = (150, 250)
    # cars_point = (175, 105)
    state = np.array([
            0.0,
            0.0,
        ], dtype=np.float32)
    has_supply = False
    rl_action = None
    actions = None
    aim_flag = False
    run = False
    rl_report_flag = True
    turret_align_flag = False
    goal_flag = False
    car_flag = True
    human_flag = True
    rl_fire = False
    path = []
    tank_cnt_list = deque([0]*5, maxlen=5)
    car_cnt_list = deque([0]*3, maxlen=3)
    human_cnt_list = deque([0]*3, maxlen=3)
    while True:
        try:
            init_data = init_input_q.get_nowait()
            print("init 데이터 수신됨:", init_data)
        except queue.Empty:
            init_data = None
        if init_data:
            clear_queue(action_input_queue, action_output_queue,
                        hit_input_queue, detect_input_queue,
                        info_input_queue, info_output_queue,
                        init_input_queue, collision_input_queue,
                        obstacles_input_queue, rl_output_q, target_point_q)
            state = np.array([
                0.0,
                0.0,
            ], dtype=np.float32)
            has_supply = False
            target_point = None
            rl_action = None
            actions = None
            aim_flag = False
            run = False
            rl_report_flag = True
            turret_align_flag = False
            goal_flag = False
            car_flag = True
            human_flag = True
            rl_fire = False
            path = []
            enemy_point = (235.81, 6.60, 290.74)
            tank_cnt_list = deque([0]*5, maxlen=5)
            car_cnt_list = deque([0]*3, maxlen=3)
            human_cnt_list = deque([0]*3, maxlen=3)
            info_output_q.put({"status": "success", "control": ""})
            continue

        log_data = info_input_q.get()
        if not log_data:
            print("로그 데이터 없음")
            info_output_q.put({"status": "success", "control": ""})
            continue

        try:
            hit_data = hit_input_q.get_nowait()
            print("포탄 충돌 정보 수신됨")
        except queue.Empty:
            hit_data = None
        if hit_data:
            print(f"포탄 충돌 정보: {hit_data}")
            if 'Tank' in hit_data['hit']:
                print("Agent: 적 탱크 격파에 성공했습니다.")
                command = {"user": "적 탱크 격파에 성공했습니다."}
                turret_align_flag = True
                aim_flag = False
                rl_fire = False
                run = False
                detection_flag = True
                # rl_report_flag = True
                state[0] = 0.0
                enemy_point = None
            clear_queue(detect_input_queue)

        try:
            collision_data = collision_input_q.get_nowait()
            print("충돌 정보 수신됨:", collision_data)
        except queue.Empty:
            collision_data = None
        if collision_data:
            print(f"충돌 정보: {collision_data}")
            
        try:
            obstacles_data = obstacles_input_q.get_nowait()
        except queue.Empty:
            obstacles_data = None
        if obstacles_data:
            astar.update_obstacle(obstacles_data)

        try:
            action_request = action_input_q.get_nowait()
        except queue.Empty:
            action_request = None
        if not action_request:
            info_output_q.put({"status": "success", "control": ""})
            continue

        # if rl_report_flag and target_point is None and goal_flag is False:
        if rl_report_flag and goal_flag is False:
            rl_action = rl.select_action(state)
            print(f"RL Action: {rl_action}")
            rl_report_flag = False

        tank_pos = (int(log_data.get("playerPos", {}).get("x")), int(log_data.get("playerPos", {}).get("z")))
        
        actions = None
        if rl_action == 0: # supply
            print("RL: 보급 획득 명령")
            rl_output_q.put({"bot": "보급을 획득하라."})
            if has_supply:
                print("Agent: 이미 보급을 획득했습니다.")
                command = {"user": "이미 보급을 획득했습니다."}
                rl_report_flag = True
                target_point = None
            else:
                target_point = supply_point
                actions = astar.get_action(log_data, target_point)
                print("Agent: 보급 획득을 위해 이동하겠습니다.")
                command = {"user": "보급 획득을 위해 이동하겠습니다."}
            rl_action = None

        elif rl_action == 1: # base
            print("RL: 기지로 복귀하라.")
            rl_output_q.put({"bot": "기지로 복귀하라."})
            if not has_supply:
                print("Agent: 현재 보급을 획득하지 못했습니다.")
                command = {"user": "보급 획득을 위해 이동하겠습니다."}
                rl_report_flag = True
                target_point = None
            else:
                target_point = base_point
                actions = astar.get_action(log_data, target_point)
                print("Agent: 기지로 복귀하겠습니다.")
                command = {"user": "기지로 복귀하겠습니다."}
            rl_action = None

        elif rl_action == 2: # fire
            print("RL: 적 탱크를 격파하라.")
            rl_output_q.put({"bot": "적 탱크를 격파하라."})
            if state[0] == -1.0:
                print("Agent: 적 탱크가 다수입니다.")
                command = {"user": "적 탱크가 다수입니다."}
                rl_report_flag = True
                target_point = None
            elif state[0] == 0.0:
                print("Agent: 적 탱크가 없습니다.")
                command = {"user": "적 탱크가 없습니다."}
                rl_report_flag = True
                target_point = None
            elif state[0] == 1.0:
                print("Agent: 적 탱크를 격파하겠습니다.")
                command = {"user": "적 탱크를 격파하겠습니다."}
                rl_fire = True
            rl_action = None
        elif rl_action == 3: # detour
            print("RL: 적을 우회하라.")
            rl_output_q.put({"bot": "적을 우회하라."})
            run = True
            target_point = None
            rl_action = None
            aim_flag = False
            turret_align_flag = True
        elif target_point:
            distance = np.hypot(target_point[0] - tank_pos[0], target_point[1] - tank_pos[1])
            if distance < 6:
                print(f"Agent: 목표 지점에 도착했습니다: {target_point}")
                command = {"user": f"목표 지점에 도착했습니다: {target_point}"}
                target_point = None
                rl_report_flag = True
            else:
                actions = astar.get_action(log_data, target_point)
        if run:
            distance = np.hypot(enemys_point[0] - tank_pos[0], enemys_point[1] - tank_pos[1])
            if distance >= 100:
                state[0] = 0.0
                print("Agent: 적을 우회했습니다.")
                command = {"user": "적을 우회했습니다."}
                astar.initial_obstacles.append({
                            "x": enemys_point[0],
                            "z": enemys_point[1],
                            "radius": 90
                })
                run = False
                target_point = None
                rl_report_flag = True
            elif target_point is None:
                def find_best_position(astar_grid, enemys_point, tank_pos, min_dist=110):
                    h, w = astar_grid.shape
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    obstacle_mask = (astar_grid == 1)
                    dist_to_enemy = np.hypot(grid_x - enemys_point[0], grid_y - enemys_point[1])
                    safe_mask = dist_to_enemy >= min_dist
                    valid_mask = (~obstacle_mask) & safe_mask
                    if not np.any(valid_mask):
                        return None
                    dist_to_tank = np.hypot(grid_x - tank_pos[0], grid_y - tank_pos[1])
                    min_idx = np.argmin(np.where(valid_mask, dist_to_tank, np.inf))
                    y, x = np.unravel_index(min_idx, astar_grid.shape)
                    return int(x), int(y)
                target_point = find_best_position(astar.pathfinder.grid, enemys_point, tank_pos)
                print("우회 좌표:", target_point)
                actions = astar.get_action(log_data, target_point)
                


        if not has_supply:
            distance = np.hypot(supply_point[0] - tank_pos[0], supply_point[1] - tank_pos[1])
            if distance < 6:
                print("Agent: 보급을 획득했습니다.")
                command = {"user": "보급을 획득했습니다."}
                has_supply = True
                rl_report_flag = True
                target_point = None
                state[1] = 1.0
        if not goal_flag and has_supply:
            distance = np.hypot(base_point[0] - tank_pos[0], base_point[1] - tank_pos[1])
            if distance < 6:
                print("Agent: 기지에 도착했습니다.")
                command = {"user": "기지에 도착했습니다."}
                goal_flag = True
                rl_report_flag = True
                target_point = None

        if actions is None:
            actions = [0.0, 0.0]  # 기본값 설정
        if target_point:
            tmp_path = astar.path
            if len(tmp_path) > 1:
                if path == tmp_path:
                    target_point_q.put({"target_point":target_point})
                else:
                    path = tmp_path
                    target_point_q.put({"target_point":target_point, "path":path})
        # 탱크 개인인지 군집인지 체크
        counters = {class_id: 0 for class_id in target_classes.keys()}
        tank_cnt = 0
        car_cnt = 0
        human_cnt = 0
        try:
            detections = detect_input_q.get_nowait()
        except queue.Empty:
            pass
        if detections:
            for box in detections:
                class_id = int(box[5])
                if class_id in [0, 1, 2] and box[4] > 0.8:
                    clear_queue(detect_input_q)
                    counters[class_id] += 1 # {0:0, 1:0, 2:0}
            tank_cnt = counters[0]
            car_cnt = counters[1]
            human_cnt = counters[2]
            tank_cnt_list.append(tank_cnt)
            car_cnt_list.append(1 if car_cnt else 0)
            human_cnt_list.append(1 if human_cnt else 0)
            
            # tank_cnt_list의 최대값 == 최빈값일 경우에 보고
            # ex) 0, 0, 2, 2, 4 -> 보고 x 0, 0, 4, 4, 1 -> 보고 x 0, 4, 4, 3, 2 -> 보고 o
            # print(f"탱크: {tank_cnt_list}, 차량: {car_cnt_list}, 사람: {human_cnt_list}")
            if sum(tank_cnt_list) and not turret_align_flag:
                
                tank_cnt_max = max(tank_cnt_list)
                if (tank_cnt_mode := Counter(tank_cnt_list).most_common()[0][0]) >= 1:
                    if tank_cnt_mode == tank_cnt_max:
                        if tank_cnt_mode > 1 and not run:
                            aim_flag = True
                            print("Agent: 적 탱크 다수 발견했습니다.")
                            command = {"user": "적 탱크 다수 발견했습니다."}
                            state[0] = -1.0
                            rl_report_flag = True
                        elif not rl_fire and not run:
                            aim_flag = True
                            print("Agent: 적 탱크 한 대 발견했습니다.")
                            command = {"user": "적 탱크 한 대 발견했습니다."}
                            state[0] = 1.0
                            rl_report_flag = True
                        # target_point = None    

            elif sum(car_cnt_list) >= 3 and not rl_fire and car_flag: # Car 보고
                car_flag = False
                target_point = cars_point
                print("Agent: 미확인 차량 발견했습니다. 해당 근처로 가서 정찰하겠습니다.")
                command = {"user": " 미확인 차량 발견했습니다. 해당 근처로 가서 정찰하겠습니다."}

            elif sum(human_cnt_list) >= 3 and not rl_fire and human_flag: # Human 보고 > llm에서 명령이 떨어지면 이동 
                human_flag = False
                print("Agent: 미확인 사람 발견했습니다. 해당 근처로 가서 정찰하겠습니다.")
                command = {"user": " 미확인 차량 발견했습니다. 해당 근처로 가서 정찰하겠습니다."}
        else: # 디텍션 안 됐을 경우
            tank_cnt_list.append(0)
            car_cnt_list.append(0)
            human_cnt_list.append(0)

        if goal_flag:
            actions[0] = -10.0
        if actions[0] > 0:
            movews = "W"
        elif actions[0] > -0.9:
            movews = "S"
        else:
            movews = "STOP"
        
        t_actions = None
        if aim_flag:
            t_actions = aim_bot.get_action(log_data, enemy_point)
        if t_actions:
            if rl_fire and t_actions[2]:
                fire = True
            else:
                fire = False
            action = {
                "moveWS": {"command": movews, "weight": abs(actions[0])},
                "moveAD": {"command": "A" if actions[1] > 0 else "D", "weight": abs(actions[1])},
                "turretQE": {"command": "Q" if t_actions[0] > 0 else "E", "weight": abs(t_actions[0]) * 0.8},
                "turretRF": {"command": "R" if t_actions[1] > 0 else "F", "weight": abs(t_actions[1]) * 0.8},
                "fire": fire
            }
        else:
            action = {
                "moveWS": {"command":movews, "weight": abs(actions[0])},
                "moveAD": {"command": "A" if actions[1] > 0 else "D", "weight": abs(actions[1])},
            }
        if turret_align_flag:
            x = np.clip(((log_data.get("playerTurretX", 0.0) - log_data.get("playerBodyX", 0.0) + 180) % 360 - 180), -0.5, 0.5)
            t_actions = [x, 0, 0]
            if abs((log_data.get("playerTurretX", 0.0) - log_data.get("playerBodyX", 0.0) + 180) % 360 - 180) <= 1:
                turret_align_flag = False
                
            else:
                action = {
                    "moveWS": {"command": movews, "weight": abs(actions[0])},
                    "moveAD": {"command": "A" if actions[1] > 0 else "D", "weight": abs(actions[1])},
                    "turretQE": {"command": "Q" if t_actions[0] > 0 else "E", "weight": abs(t_actions[0])},
                    "turretRF": {"command": "R" if t_actions[1] > 0 else "F", "weight": abs(t_actions[1])},
                    "fire": ""
                }
        action_output_q.put(action)
        if command:
            rl_output_q.put(command)
            command = None
        info_output_q.put({"status": "success", "control": ""})

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    pil_image = Image.open(BytesIO(image.read()))
    yolo_input_queue.put(pil_image)
    try:
        detections = yolo_output_queue.get(timeout=1)
    except queue.Empty:
        return jsonify({})
    detect_input_queue.put(detections)
    filtered_results = []
    # for box in detections:
    #     class_id = int(box[5])
    #     if class_id in target_classes and box[4] > 0.8:
    #         filtered_results.append({
    #             'className': target_classes[class_id],
    #             'bbox': [float(coord) for coord in box[:4]],
    #             'confidence': float(box[4]),
    #             'color': '#00FF00',
    #             'filled': False,
    #             'updateBoxWhileMoving': False
    #         })
    return jsonify(filtered_results)

@app.route('/tank_status', methods=['GET'])
def tank_status():
    return jsonify(tank_status_data)
def calculate_impact_point_on_ground(x, y, z, yaw_deg, pitch_deg, gravity=9.81):
    initial_speed = 54
    turret_length = 5.891
    turret_offset = turret_length / 2
    # y 좌표는 포탑 높이를 고려하여 조정
    y -= 5
    # yaw를 라디안으로 변환
    yaw = math.radians(yaw_deg)
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
    distance = get_distance_for_pitch(pitch_deg)
    impact_x = x + turret_offset * math.sin(yaw) + distance * math.sin(yaw)
    impact_z = z + turret_offset * math.cos(yaw) + distance * math.cos(yaw)
    
    return (impact_x, impact_z)


# GOOGLE_API_KEY = "AIzaSyAG6S4DQtZlHbIxBQsHp9Ab_Bek7SPMSgY"  # 여기에 Gemini API 키 입력
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-2.5-flash')
# chat = model.start_chat(history=[])
@app.route('/info', methods=['POST'])
def info():
    command = None
    try:
        command = rl_output_queue.get_nowait()
        print(command)
        if "user" in command:
            tank_status_data.update({
                "user_command":command["user"]
            })
            chat_history.append(("User", command))
        elif "bot" in command:
            tank_status_data.update({
                "bot_command":command["bot"]
                })
            chat_history.append(("Bot", command))
    except queue.Empty:
        pass
        
    

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    info_input_queue.put(data)
    x = data.get("playerPos", {}).get("x")
    y = data.get("playerPos", {}).get("y")
    z = data.get("playerPos", {}).get("z")
    yaw = data.get("playerTurretX")
    pitch = data.get("playerTurretY")
    # impact_data = calculate_impact_point_on_ground(x, z, yaw, pitch)
    impact_data = calculate_impact_point_on_ground(x, y, z, yaw, pitch)
    try:
        target_point = target_point_queue.get_nowait()
    except queue.Empty:
        target_point = None
    target_x, target_z = None, None
    if target_point:
        if "target_point" in target_point:
            target_x, target_z = target_point["target_point"]
            tank_status_data.update({
                "goal_x":target_x,
                "goal_z":target_z
            })
        if "path" in target_point:
            tank_status_data.update({
                "path":target_point["path"]
            })
    tank_status_data.update({
        "x": x,
        "z": z,
        "yaw": yaw,
        "pitch": pitch,
        "impact_x": impact_data[0],
        "impact_z": impact_data[1],
    })
    global player_lidar_angle_z, lidar_data, player_pos
    player_lidar_angle_z = {'z': data.get("lidarRotation", [])['y']}
    lidar_data_raw = data.get("lidarPoints", [])
    player_pos = {'x': data.get("playerPos", [])['x'], 'z': data.get("playerPos", [])['z']}
    lidar_data = []
    for point in lidar_data_raw:
        if point.get('channelIndex') == 3:
            angle = point.get('angle')
            pos = point.get('position', {})
            lidar_data.append({
                'angle': angle,
                'x': pos.get('x'),
                'z': pos.get('z')
            })

    try:
        response = info_output_queue.get(timeout=1)
    except queue.Empty:
        response = {}
    return jsonify(response)

@app.route('/get_action', methods=['POST'])
def get_action():
    action_input_queue.put(True)
    try:
        action = action_output_queue.get(timeout=1)
    except queue.Empty:
        action = {}
    return jsonify(action)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    hit_input_queue.put(data)
    tank_status_data.update({
        "real_impact_x": data.get('x'),
        "real_impact_z": data.get('z')
    })
    time.sleep(1)
    tank_status_data.update({
        "real_impact_x": 0,
        "real_impact_z": 0
    })
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/minimap')
def minimap():
    return render_template("index.html")

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/obstacles')
def get_obstacles():
    return jsonify(obstacle_data)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data.get("obstacles"):
        return jsonify({'status': 'error', 'message': 'No data received'}), 400
    obstacles_input_queue.put(data)
    global obstacles
    obstacles = data['obstacles']
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST'])
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400
    collision_input_queue.put(data)
    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')
    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "start",
        "blStartX": 8,
        "blStartY": 10,
        "blStartZ": 290,
        "rdStartX": 180,
        "rdStartY": -10,
        "rdStartZ": 60,
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    init_input_queue.put(True)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

@app.route('/data')
def data():
    return jsonify({
        'obstacles': obstacles,
        'lidar': lidar_data,
        'playerPos': player_pos,
        'playerLidarAngleZ': player_lidar_angle_z,

    })


@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

if __name__ == '__main__':
    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # 불필요한 로그 감소
    yolo_proc = Process(target=yolo_worker, args=(yolo_input_queue, yolo_output_queue))
    action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue,
                                                      hit_input_queue, detect_input_queue,
                                                      info_input_queue, info_output_queue,
                                                      init_input_queue, collision_input_queue,
                                                      obstacles_input_queue, rl_output_queue, target_point_queue
                                                      ))
    yolo_proc.start()
    action_proc.start()
    app.run(host='0.0.0.0', port=5036, threaded=True, debug=False)
