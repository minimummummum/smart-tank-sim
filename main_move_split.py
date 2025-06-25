from flask import Flask, request, jsonify, render_template
from multiprocessing import Process, Queue
from ultralytics import YOLO
from PIL import Image
from collections import Counter
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

# # flat map test -> 도착 지점 (80, 265)
# CONFIG = {
#         "startMode": "start",  # Options: "start" or "pause"
#         "blStartX": 80,  #Blue Start Position
#         "blStartY": 10,
#         "blStartZ": 32,
#         "rdStartX": 144, #Red Start Position (146, 36)
#         "rdStartY": 10,
#         "rdStartZ": 34,
#         "trackingMode": True,
#         "detactMode": True,
#         "logMode" :True,
#         "enemyTracking": False,
#         "saveSnapshot": False,
#         "saveLog": False,
#         "saveLidarData": False,
#         "lux": 30000
#     }

# # S1 map -> 도착 지점 (207, 56)
# CONFIG = {
#         "startMode": "start",  # Options: "start" or "pause"
#         "blStartX": 20,  #Blue Start Position
#         "blStartY": 10,
#         "blStartZ": 280, 
#         "rdStartX": 200, #Red Start Position 
#         "rdStartY": 10,
#         "rdStartZ": 50,
#         "trackingMode": True,
#         "detactMode": True,
#         "logMode" :True,
#         "enemyTracking": False,
#         "saveSnapshot": False,
#         "saveLog": False,
#         "saveLidarData": False,
#         "lux": 30000
#     }

# S2 map -> 도착 지점 (20, 280)
CONFIG = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 207,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 56, 
        "rdStartX": 211, #Red Start Position (146, 36)
        "rdStartY": 10,
        "rdStartZ": 240,
        "trackingMode": True,
        "detactMode": True,
        "logMode" :True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }

app = Flask(__name__)
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
    "real_impact_z": 0
}
obstacle_data = {}
chat_history = []  # 채팅 기록 저장 리스트
tank_cnt_list = []  # 적 탱크의 수를 확실히 하기 위한 리스트

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

target_classes = {0: "Car", 3: "E_Tank", 4: "Human"}        # 2500n.pt
# target_classes = {0: "E_Tank", 1: "Car", 2: "Human"}      # xmodel_test.pt OR lmodel_test.pt
# target_classes = {0: "Car", 1: "Rock", 2: "Wall", 3: "E_Tank", 4: "Human", 5: "Mine"}
tank_id = next(key for key, value in target_classes.items() if value == "E_Tank")

def yolo_worker(yolo_input_q, yolo_output_q):
    model = YOLO("2500n.pt").to("cuda")       # nano model test용
    # model = YOLO("lmodel_test.pt").to("cuda")       # large model test용
    # model = YOLO("xmodel_test.pt").to("cuda")       # xlarge model test용

    # YOLO 프로세스 반복
    while True:
        # /detect request yolo_input_q에서 이미지 가져오기
        image = yolo_input_q.get()
        results = model(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy().tolist()
        # YOLO 결과를 yolo_output_q에 넣어 /detect로 response
        yolo_output_q.put(detections)


def action_worker(action_input_q, action_output_q, hit_input_q, detect_input_q,
                  info_input_q, info_output_q, init_input_q, collision_input_q, obstacles_input_q, llm_input_q):
    hit_data = None
    detections = None
    init_data = None
    collision_data = None
    astar = pf.Path()
    aim_bot = aim.Aim()
    target_point = None
    actions = None
    speed_flag = False      # 감속
    fire_flag = False       # 발사
    stop_flag= False        # 정지
    global tank_cnt_list

    while True:
        counters = {class_id: 0 for class_id in target_classes.keys()}
        tank_cnt=0
        try:
            init_data = init_input_q.get_nowait()
            print("init 데이터 수신됨:", init_data)
        except queue.Empty:
            init_data = None
        if init_data:
            clear_queue(action_input_queue, action_output_queue,
                        hit_input_queue, detect_input_queue,
                        info_input_queue, info_output_queue,
                        init_input_queue, collision_input_queue, obstacles_input_queue, llm_input_queue)
            info_output_q.put({"status": "success", "control": ""})
            target_point=None
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
            pass

        try:
            collision_data = collision_input_q.get_nowait()
            print("충돌 정보 수신됨:", collision_data)
        except queue.Empty:
            collision_data = None
        if collision_data:
            print(f"충돌 정보: {collision_data}")
            pass

        try:
            obstacles_data = obstacles_input_q.get_nowait()
            print("장애물 정보 수신됨:", obstacles_data)
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
            x, y = map(int, llm_request.strip("()").split(","))
            llm_request = None
            target_point = [x, y]
            print(target_point)
            speed_flag = False
            fire_flag = False
            stop_flag = False
            clear_queue(detect_input_q)
        if target_point:
            actions = astar.get_action(log_data, target_point)
        if actions is None:
            continue


        try:
            detections = detect_input_q.get_nowait()
        except queue.Empty:
            detections = None
        if detections:
            # tank_cnt : 한 프레임에서 detect된 탱크 수
            for box in detections:
                class_id = int(box[5])
                if class_id in target_classes.keys() and box[4] > 0.80:
                    print("객체 감지")
                    speed_flag = True
                    clear_queue(detect_input_q)
                    
                    # 각 객체의 갯수 세기
                    counters[class_id] += 1
            tank_cnt=counters[tank_id]

            tank_cnt_list.append(tank_cnt)

            if len(tank_cnt_list) >= 3:
                # tank_cnt_list의 길이는 3으로 유지 -> tank_cnt_list의 첫번째 요소 제거
                if len(tank_cnt_list) > 3:
                    tank_cnt_list.pop(0)
                # 최종 탱크 수는 tank_cnt_list의 최댓값
                tank_cnt = max(tank_cnt_list)
                # 만약 리스트 내에 0이 제일 많으면 최종 탱크 수는 0
                if Counter(tank_cnt_list).most_common()[0][0]==0:
                    tank_cnt=0
                    
        print("🚗", tank_cnt)

        # 탱크 0대 발견 시, 변화 없이 진행
        if tank_cnt == 0:
            speed_flag = False
            stop_flag = False
            fire_flag = False

        # 탱크 한 대 발견 시, 감속 
        elif tank_cnt == 1:
            speed_flag = True
            stop_flag = False
            fire_flag = False

        # 탱크 2대 이상 발견 시, 정지 
        elif tank_cnt >= 2:
            speed_flag = False
            stop_flag = True
            fire_flag = False
        
    # moveWS, moveAD 조절 
        if speed_flag:
            if log_data.get("playerSpeed", 0.0) > 5:
                actions[0] = -0.85
            # actions[0] = -10.0 # 정지

        if stop_flag:
            actions[0] = -10.0
        print(actions)

        if actions[0] > 0:
            movews = "W"
        elif actions[0] > -0.9:
            movews = "S"
        else:
            movews = "STOP"

    # turretQE, turretRF 조절
        if fire_flag:
            t_actions = aim_bot.get_action(log_data)
        else:
            t_actions = [0, 0, 0]

    # 최종 action 결정
        if t_actions:
            action = {
                "moveWS": {"command": movews, "weight": abs(actions[0])},
                "moveAD": {"command": "A" if actions[1] > 0 else "D", "weight": abs(actions[1])},
                "turretQE": {"command": "Q" if t_actions[0] > 0 else "E", "weight": abs(t_actions[0])},
                "turretRF": {"command": "R" if t_actions[1] > 0 else "F", "weight": abs(t_actions[1])},
                "fire": bool(t_actions[2])
            }
        else:
            action = {
                "moveWS": {"command": movews, "weight": abs(actions[0])},
                "moveAD": {"command": "A" if actions[1] > 0 else "D", "weight": abs(actions[1])},
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
        if class_id in target_classes and box[4] > 0.80:
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

def calculate_impact_point_on_ground(x, z, yaw_deg, pitch_deg, gravity=9.81):
    initial_speed = 60
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    dx = math.cos(pitch) * math.sin(yaw)
    dy = math.sin(pitch)
    dz = math.cos(pitch) * math.cos(yaw)
    v0x = initial_speed * dx
    v0y = initial_speed * dy
    v0z = initial_speed * dz
    if v0y <= 0:
        return (x, 0, z)
    t_impact = 2 * v0y / gravity
    impact_x = x + v0x * t_impact
    impact_z = z + v0z * t_impact
    return (impact_x, impact_z)

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    info_input_queue.put(data)
    x = data.get("playerPos", {}).get("x")
    y = data.get("playerPos", {}).get("y")
    z = data.get("playerPos", {}).get("z")
    yaw = data.get("playerTurretX")
    pitch = data.get("playerTurretY")
    impact_data = calculate_impact_point_on_ground(x, z, yaw, pitch)
    tank_status_data.update({
        "x": x,
        "z": z,
        "yaw": yaw,
        "pitch": pitch,
        "impact_x": impact_data[0],
        "impact_z": impact_data[1]
    })
    global player_lidar_angle_z, lidar_data, player_pos
    player_lidar_angle_z = {'z': data.get("lidarRotation", [])['y']}
    lidar_data_raw = data.get("lidarPoints", [])
    player_pos = {'x': data.get("playerPos", [])['x'], 'z': data.get("playerPos", [])['z']}
    lidar_data = []
    for point in lidar_data_raw:
        if point.get('channelIndex') == 2:
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
    print("🪨 Obstacle Data:", data)
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
    config = CONFIG
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
        'playerLidarAngleZ': player_lidar_angle_z
    })

# Gemini API 키 설정 (실제 키로 교체 필요)
GOOGLE_API_KEY = "AIzaSyAjrWVzUEwwIgXlDKUnCdMxm4DVbs5g_RM"  # 여기에 Gemini API 키 입력
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
chat = model.start_chat(history=[])
@app.route('/chat', methods=['POST'])
def receive_chat():
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'user_message' not in data:
            return jsonify({"status": "error", "message": "Invalid data format"}), 400
        user_message = data['user_message']
        #try:
            #response = chat.send_message(user_message)
            #bot_response = response.text
        #except Exception as e:
            #bot_response = f"Gemini API 오류: {str(e)}"
        print(f"Flask 서버에서 받은 메시지 - 사용자: {user_message}")
        llm_input_queue.put(user_message) # bot_response -> user_message 임의로 llm 해제
        chat_history.append(("User", user_message))
        #chat_history.append(("Bot", bot_response))
        return jsonify({"status": "success", "response": user_message}), 200
    return jsonify({"status": "error", "message": "Method not allowed"}), 405

@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

if __name__ == '__main__':
    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # 불필요한 로그 감소
    yolo_proc = Process(target=yolo_worker, args=(yolo_input_queue, yolo_output_queue))
    action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue,
                                                      hit_input_queue, detect_input_queue,
                                                      info_input_queue, info_output_queue,
                                                      init_input_queue, collision_input_queue, obstacles_input_queue, llm_input_queue))
    yolo_proc.start()
    action_proc.start()
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)