from flask import Flask, request, jsonify, render_template
from multiprocessing import Process, Queue
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from collections import Counter, deque
import numpy as np
from io import BytesIO
import logging
import queue
from utils import clear_queue
import time
import math
import google.generativeai as genai
import core.path_Finder as pf 
import core.aim as aim
from collections import Counter, deque
SCENARIO = """
당신은 시뮬레이터의 자율주행 탱크에게 명령을 내리는 사령관입니다.
당신의 임무는 처음에 주어지는 시나리오를 기반으로 탱크의 보고에 따라 상황을 판단하고,
가장 적절한 다음 행동을 추론하여 제시하는 것입니다.
**규칙:**
1. **맵 사이즈:** 300x300 (좌표 범위: 0부터 299까지)
2. **시작 좌표:** (5,295)
3. **답변 형식:** 오직 다음 이동할 좌표와 사격 여부를 알려주면 됩니다. 형식은 pos:[100,100], fire:False 이며, fire는 적 인식 보고가 들어 왔을 때 가능하며, 사거리는 100이내 일 때 가능합니다.
4. **언어:** 답변은 한글로만 합니다.
5. **추론:** 당신 스스로 상황을 분석하고 최적의 좌표(이동할 최종 좌표)와 사격 여부를 선정해야 합니다.
6. **상황:** 상황은 시뮬레이터의 자율주행 탱크가 당신의 명령을 완료하거나, 적 탱크 및 다른 장애물 인식 시 보고합니다.
7. **보고:** 적탱크/자동차/사람 좌표와 몇대인지 주어집니다. 
적 탱크 발견 시 한 대일 경우 사격하고, 여러대일 경우 회피하세요.
사격할 때의 이동 좌표는 원래 최종 목적지로 주세요.
만약 회피할 경우, 좌표를 후퇴 좌표로 주지 말고 우회 좌표로 주세요.
자동차나 사람을 발견하면 거기 근처로 근접해서 정찰하세요.
이제 시뮬레이션을 시작합니다.
시나리오 시작
(280, 280)으로 이동해서 보급을 받고, 다시 복귀하라
"""
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
    "user_command":None
}
obstacle_data = {}
chat_history = []  # 채팅 기록 저장 리스트
tank_cnt_list = deque(maxlen=3)  # 적 탱크의 수를 확실히 하기 위한 리스트

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
llm_input_queue = Queue(maxsize=1)
llm_output_queue = Queue(maxsize=1)
target_point_queue = Queue(maxsize=1)

# target_classes = {0: "Car", 3: "E_Tank", 4: "Human"}
target_classes = {1: "Car", 0: "E_Tank", 2: "Human"}
def yolo_worker(yolo_input_q, yolo_output_q):
    model = YOLO("yolov8x_e500_s512_b8.pt").to("cuda")
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
                  obstacles_input_q, llm_input_q, llm_output_q, target_point_q):
    hit_data = None
    detections = None
    init_data = None
    collision_data = None
    astar = pf.Path()
    aim_bot = aim.Aim()
    target_point = None
    actions = None
    speed_flag = False
    aim_flag = False
    stop_flag = False
    llm_report_flag = True
    turret_align_flag = False
    goal_flag = False
    car_flag = True
    tank_cnt_list = deque([0]*5, maxlen=5)
    car_cnt_list = deque([0]*5, maxlen=5)
    human_cnt_list = deque([0]*5, maxlen=5)
    llm_output_q.put(SCENARIO)
    while True:
        try:
            init_data = init_input_q.get_nowait()
            print("init 데이터 수신됨:", init_data)
        except queue.Empty:
            init_data = None
        if init_data:
            # clear_queue(action_input_queue, action_output_queue,
            #             hit_input_queue, detect_input_queue,
            #             info_input_queue, info_output_queue,
            #             init_input_queue, collision_input_queue,
            #             obstacles_input_queue, llm_input_queue,
            #             llm_output_q, target_point_queue)
            clear_queue(action_input_queue, action_output_queue,
                        hit_input_queue, detect_input_queue,
                        info_input_queue, info_output_queue,
                        init_input_queue, collision_input_queue,
                        obstacles_input_queue, llm_input_queue,
                        llm_output_q, target_point_q)
            tank_cnt_list = deque([0]*5, maxlen=5)
            car_cnt_list = deque([0]*5, maxlen=5)
            human_cnt_list = deque([0]*5, maxlen=5)
            info_output_q.put({"status": "success", "control": ""})
            continue

        log_data = info_input_q.get()
        if not log_data:
            print("로그 데이터 없음")
            info_output_q.put({"status": "success", "control": ""})
            continue
        else:
        # detection 없고 터렛의 방향이 몸체 방향과 일치하면 터렛 aim 움직이지 않게 하기 위해서
        # 현재 터렛 방향과 몸체 방향의 angle_diff 계산
            # 포탑 각도
            tur_x = log_data.get("playerTurretX")
            # 몸체 각도
            body_x= log_data.get("playerBodyX")
            angle_diff = ((tur_x - body_x + 180) % 360) - 180


        try:
            hit_data = hit_input_q.get_nowait()
            print("포탄 충돌 정보 수신됨")
        except queue.Empty:
            hit_data = None
        if hit_data:
            print(f"포탄 충돌 정보: {hit_data}")
            if 'Tank' in hit_data['hit']:
                turret_align_flag = True
                aim_flag = False
                llm_fire = False
                speed_flag = False
                stop_flag = False
                llm_report_flag = True
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

        try:
            llm_request = llm_input_q.get_nowait()
        except queue.Empty:
            llm_request = None
        if not llm_request:
            pass
       
        if llm_request:
            llm_pos, llm_fire = llm_request
            llm_request = None
            target_point = llm_pos
            if target_point:
                target_point_q.put(target_point)
            print(f"Action worker received new LLM target: {target_point}")
            speed_flag = False
            # aim_flag = False
            stop_flag = False
            clear_queue(detect_input_q)
        tank_pos = (int(log_data.get("playerPos", {}).get("x")), int(log_data.get("playerPos", {}).get("z")))
        if target_point:
            actions = astar.get_action(log_data, target_point)
            distance = np.hypot(target_point[0] - tank_pos[0], target_point[1] - tank_pos[1])

            if distance < 5 and not goal_flag and not llm_report_flag:
                goal_flag = True
                llm_report_flag = True
            if distance > 10 and goal_flag and not llm_report_flag:
                goal_flag = False
                llm_report_flag = True
            if distance < 5 and not turret_align_flag and not aim_flag and llm_report_flag:# and not goal_flag:
                llm_output_q.put(f"{target_point}에 도착했습니다.")
                goal_flag = True
                llm_report_flag = False
            
        if actions is None:
            continue
        
        
        # 탱크 개인인지 군집인지 체크
        counters = {class_id: 0 for class_id in target_classes.keys()}
        tank_cnt = 0
        car_cnt = 0
        human_cnt = 0
        try:
            detections = detect_input_q.get_nowait()
        except queue.Empty:
            detections = None
        if detections:
            for box in detections:
                class_id = int(box[5])
                if class_id in [0, 1, 2] and box[4] > 0.8:
                    speed_flag = True
                    clear_queue(detect_input_q)
                    counters[class_id] += 1 # {0:0, 1:0, 2:0}
                else:
                    speed_flag = False
            tank_cnt = counters[0]
            car_cnt = counters[1]
            human_cnt = counters[2]
            tank_cnt_list.append(tank_cnt)
            car_cnt_list.append(1 if car_cnt else 0)
            human_cnt_list.append(1 if human_cnt else 0)
            
            # tank_cnt_list의 최대값 == 최빈값일 경우에 보고
            # ex) 0, 0, 2, 2, 4 -> 보고 x 0, 0, 4, 4, 1 -> 보고 x 0, 4, 4, 3, 2 -> 보고 o
            if sum(tank_cnt_list) and not turret_align_flag:
                
                tank_cnt_max = max(tank_cnt_list)
                if (tank_cnt_mode := Counter(tank_cnt_list).most_common()[0][0]) >= 1:
                    if tank_cnt_mode == tank_cnt_max and llm_report_flag:
                        if tank_cnt_mode > 1:
                            stop_flag = True
                            print(tank_cnt_mode, "탱크 발견 수")
                            aim_flag = False
                        else:
                            aim_flag = True
                            speed_flag = True
                            stop_flag = True
                            # 적 탱크 1대 or 여러대 발견으로 수정
                        tank_cnt_mode = 1 if tank_cnt_mode == 1 else "여러"
                        llm_output_q.put(f'{target_point}로 가던 도중 {tank_pos}까지 왔는데, 적 탱크 {tank_cnt_mode}대 발견! 조준 완료. 격파 여부 대기')
                        llm_report_flag = False
                        
            elif sum(car_cnt_list) >= 3 and llm_report_flag and car_flag: # Car 보고
                # llm_report_flag = False 
                car_flag = False
                stop_flag = True
                llm_output_q.put(f'현재 좌표: {tank_pos}, (137, 100) 에서 차 발견')
            elif sum(human_cnt_list) >= 3 and llm_report_flag: # Human 보고 > llm에서 명령이 떨어지면 이동 
                llm_report_flag = False
                print(f"사람 발견! 명령 대기") # 팝업 창
                #llm_output_q.put(f'(280, 150)에서 사람 발견') # 현재 좌표 ({log_data.get("playerPos", {}).get("x")}, {log_data.get("playerPos", {}).get("z")}), 
        else: # 디텍션 안 됐을 경우
            tank_cnt_list.append(0)
            car_cnt_list.append(0)
            human_cnt_list.append(0)
            speed_flag = False    


        # if speed_flag:
        #     if log_data.get("playerSpeed", 0.0) > 2:
        #         actions[0] = -0.85


        if stop_flag:
            actions[0] = -10.0

        if actions[0] > 0:
            movews = "W"
        elif actions[0] > -0.9:
            movews = "S"
        else:
            movews = "STOP"

        ################
        
        t_actions = None
        if aim_flag:
            t_actions = aim_bot.get_action(log_data)
        if t_actions:
            if llm_fire == t_actions[2] == True:
                fire = True
            else:
                fire = False
            action = {
                "moveWS": {"command": movews, "weight": abs(actions[0])},
                "moveAD": {"command": "A" if actions[1] > 0 else "D", "weight": abs(actions[1])},
                "turretQE": {"command": "Q" if t_actions[0] > 0 else "E", "weight": abs(t_actions[0])},
                "turretRF": {"command": "R" if t_actions[1] > 0 else "F", "weight": abs(t_actions[1])},
                "fire": fire
            }
        else:
            action = {
                "moveWS": {"command": movews, "weight": abs(actions[0])},
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
                    "fire": fire
                }
        action_output_q.put(action)
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
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes and box[4] > 0.8:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })
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
# def calculate_impact_point_on_ground(x, z, yaw_deg, pitch_deg, gravity=9.81):
#     initial_speed = 60
#     yaw = math.radians(yaw_deg)
#     pitch = math.radians(pitch_deg)
#     dx = math.cos(pitch) * math.sin(yaw)
#     dy = math.sin(pitch)
#     dz = math.cos(pitch) * math.cos(yaw)
#     v0x = initial_speed * dx
#     v0y = initial_speed * dy
#     v0z = initial_speed * dz
#     if v0y <= 0:
#         return (x, 0, z)
#     t_impact = 2 * v0y / gravity
#     impact_x = x + v0x * t_impact
#     impact_z = z + v0z * t_impact
#     return (impact_x, impact_z)

GOOGLE_API_KEY = "AIzaSyAG6S4DQtZlHbIxBQsHp9Ab_Bek7SPMSgY"  # 여기에 Gemini API 키 입력
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
chat = model.start_chat(history=[])
@app.route('/info', methods=['POST'])
def info():
    llm_input_data = None
    try:
        llm_input_data = llm_output_queue.get_nowait()
        tank_status_data.update({
            "user_command":llm_input_data
        })
        print("보고", llm_input_data)
    except queue.Empty:
        pass
    if llm_input_data:
        user_message = llm_input_data
        try:
            response = chat.send_message(user_message)
            bot_response = response.text
            pos_start = bot_response.find('[') + 1
            pos_end = bot_response.find(']')
            pos_str = bot_response[pos_start:pos_end]  # "20,280"
            pos = [int(x.strip()) for x in pos_str.split(',')]  # [20, 280]
            fire_start = bot_response.find('fire:') + len('fire:')
            fire_str = bot_response[fire_start:].split('}')[0].strip()  # "False"
            fire = True if fire_str == 'True' else False
            print("pos:", pos)
            print("fire:", fire)
            llm_input_queue.put((pos, fire))
            command = None
            if fire:
                command = f"적 격파 후 {pos}로 이동하라"
            else:
                command = f"{pos}로 이동하라"
            tank_status_data.update({
            "bot_command":command
            })
        except Exception as e:
            bot_response = f"Gemini API 오류: {str(e)}"
        print(f"LLM Agent: {bot_response}")
        chat_history.append(("User", user_message))
        chat_history.append(("Bot", bot_response))
    

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
        target_x, target_z = target_point
        tank_status_data.update({
            "goal_x":target_x,
            "goal_z":target_z
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
        "blStartX": 5,
        "blStartY": 10,
        "blStartZ": 295,
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
    # action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue,
    #                                                   hit_input_queue, detect_input_queue,
    #                                                   info_input_queue, info_output_queue,
    #                                                   init_input_queue, collision_input_queue,
    #                                                   obstacles_input_queue, llm_input_queue,
    #                                                   llm_output_queue, target_point_queue
    #                                                   ))
    action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue,
                                                      hit_input_queue, detect_input_queue,
                                                      info_input_queue, info_output_queue,
                                                      init_input_queue, collision_input_queue,
                                                      obstacles_input_queue, llm_input_queue,
                                                      llm_output_queue, target_point_queue
                                                      ))
    yolo_proc.start()
    action_proc.start()
    app.run(host='0.0.0.0', port=5036, threaded=True, debug=False)
