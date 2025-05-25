from flask import Flask, request, jsonify
from multiprocessing import Process, Queue
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import logging
import ppo_test as ppo
import queue
from utils import clear_queue
import time

app = Flask(__name__)
############################
# 1. Queue 늘리고 전역변수 없애기 (완료)
# 2. reset wait 무한 대기 -> 생각해보니 시간 이런 제약말고 /init 오면 초기화 된 거니까 그거를 시점으로
# 3. 생각을 좀 해봐야함. 현재 step()에서 왜 포탑 각도 갱신을 하는지? 시뮬레이터 상태를 불러와야지
# 초기화 상태 전송, 그거에 대한 액션 반환, 액션 실행, 상태 전송, 액션 반환, 실행..
# 초기화 -> 상태 모델 전송 -> 액션 선택 -> 시뮬레이터 액션 명령 -> 상태 모델 전송 -> 보상 처리
# 4. 보상설정 제대로 하기
############################

# 큐 생성
yolo_input_queue = Queue()
yolo_output_queue = Queue()

action_input_queue = Queue()
action_output_queue = Queue()

detect_input_queue = Queue()

hit_input_queue = Queue()

info_input_queue = Queue()
info_output_queue = Queue()

target_classes = {0: "Car", 1: "Rock", 2: "Wall", 3: "E_Tank", 4: "Human", 5: "Mine"}
# YOLO 모델 백그라운드 프로세스
def yolo_worker(input_q, output_q):
    model = YOLO("yolov8n.pt").to("cuda")
    while True:
        image = input_q.get()
        #if image is None:
            #break
        results = model(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy().tolist()
        output_q.put(detections)

# action 백그라운드 프로세스
def action_worker(action_input_q, action_output_q, hit_input_q, detect_input_q, info_input_q, info_output_q):
    info_output_q.put("reset")
    env = ppo.TankEnv()
    agent = ppo.PPOAgent(state_dim=10, action_dim=3)
    num_episodes = 1000
    hit_data = None
    detections = None
    for episode in range(num_episodes):
        reset_delay_flag = True
        memory = []
        total_reward = 0
        state = None
        env.reset()
        while True:
            # log data, action_request 받을 때까지 대기
            log_data = info_input_q.get()
            # log_data 없을 경우 
            if log_data is None:
                print("로그 데이터 없음")
                info_output_q.put({"status": "success", "control": ""})
                continue
            if reset_delay_flag:
                print("reset wait")
                if log_data.get("time") > 30:
                    continue
                else:
                    reset_delay_flag = False
            # 포탄 착탄 데이터 get
            try:
                hit_data = hit_input_q.get_nowait()
            except queue.Empty:
                hit_data = None
            # detect 데이터 get
            try:
                detections = detect_input_q.get_nowait()
            except queue.Empty:
                detections = None
            # get_action 요청
            try:
                action_request = action_input_q.get_nowait()
            except queue.Empty:
                action_request = None
            if action_request is None:
                action_output_q.put({"moveWS": {"command": "STOP", "weight": 1.0}})
                info_output_q.put({"status": "success", "control": ""})
                continue
            # 시뮬레이터 상태를 모델에 갱신
            env.update_state(log_data, hit_data)
            hit_data = None
            if state is None:
                state = env.get_state()
            action, log_prob, value = agent.select_action(state)
            sim_action = {
            "moveWS": {"command": "", "weight": 0.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "Q" if action[0] > 0.0 else "E", "weight": float(abs(action[0]))},
            "turretRF": {"command": "R" if action[1] > 0.0 else "F", "weight": float(abs(action[1]))},
            "fire": bool(action[2] > 0.0)
            }
            print("fire:", action[2])
            action_output_q.put(sim_action)
            next_state, reward, done = env.step(action)
            memory.append((state, action, log_prob, reward, value, done))
            state = next_state
            total_reward += reward
            info_output_q.put({"status": "success", "control": ""})
            # 초기화
            if done:
                info_output_q.put({"status": "success", "control": "reset"})
                clear_queue(hit_input_q, detect_input_q, action_input_q, info_output_q, info_input_q)
                print("reset!")
                time.sleep(1)
                break
        if len(memory) >= 2:
            agent.update(memory)
        else:
            print(f"메모리 길이 부족. 건너뜀. 현재 길이: {len(memory)}")
        # if memory:
        #     agent.update(memory)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            agent.save()
            print("torch 저장완료!")

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    pil_image = Image.open(BytesIO(image.read()))
    np_image = np.array(pil_image)
    
    yolo_input_queue.put(np_image)
    detections = yolo_output_queue.get()  # 결과 기다림

    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    info_input_queue.put(data)
    return jsonify(info_output_queue.get())

@app.route('/get_action', methods=['POST'])
def get_action():
    action_input_queue.put(True)
    action = action_output_queue.get()
    return jsonify(action)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()

    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400
    hit_input_queue.put(data)
    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

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

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    #x = np.random.randint(0, 300)
    #z = np.random.randint(0, 300)
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position 60
        "blStartY": 10,
        "blStartZ": 27.23, # 27.23
        "rdStartX": 59, #Red Start Position
        "rdStartY": 14,
        "rdStartZ": 150,
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
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

# 밑에 로그 안 뜨게게
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

if __name__ == '__main__':
    # 백그라운드 프로세스 시작
    yolo_proc = Process(target=yolo_worker, args=(yolo_input_queue, yolo_output_queue))
    action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue,
                                                      hit_input_queue, detect_input_queue,
                                                      info_input_queue, info_output_queue))
    yolo_proc.start()
    action_proc.start()

    app.run(host='0.0.0.0', port=5005, threaded=True)