from flask import Flask, request, jsonify
from multiprocessing import Process, Queue
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import logging
import sac_train as sac
import queue
from utils import clear_queue
import time
import math
import random
import torch
app = Flask(__name__)
############################
# 1. Queue 늘리고 전역변수 없애기 (완료)
# 2. reset wait 무한 대기 -> 생각해보니 시간 이런 제약말고 /init 오면 초기화 된 거니까 그거를 시점으로 (완료)
# 3. 생각을 좀 해봐야함. 현재 step()에서 왜 포탑 각도 갱신을 하는지? 시뮬레이터 상태를 불러와야지 (완료)
# 초기화 상태 전송, 그거에 대한 액션 반환, 액션 실행, 상태 전송, 액션 반환, 실행..
# 초기화 -> 상태 모델 전송 -> 액션 선택 -> 시뮬레이터 액션 명령 -> 상태 모델 전송 -> 보상 처리
# 4. 보상설정 제대로 하기 (완료)
# 5. PPO -> DQN으로 변경
############################

# 큐 생성
yolo_input_queue = Queue()
yolo_output_queue = Queue()

action_input_queue = Queue()
action_output_queue = Queue()

detect_input_queue = Queue()
init_input_queue = Queue()
hit_input_queue = Queue()
collision_input_queue = Queue()

info_input_queue = Queue()
info_output_queue = Queue()

target_classes = {0: "E_Tank", 1: "Car", 2: "Human"}
# YOLO 모델 백그라운드 프로세스
def yolo_worker(yolo_input_q, yolo_output_q):
    model = YOLO("ntest_2.pt").to("cuda")
    # YOLO 프로세스 반복
    while True:
        # /detect request yolo_input_q에서 이미지 가져오기
        image = yolo_input_q.get()
        results = model(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy().tolist()
        # YOLO 결과를 yolo_output_q에 넣어 /detect로 response
        yolo_output_q.put(detections)

# action 백그라운드 프로세스
def action_worker(action_input_q, action_output_q, hit_input_q, detect_input_q,
                  info_input_q, info_output_q, init_input_q, collision_input_q):
    num_episodes = 1000
    env = sac.TankEnv()
    agent = sac.SAC(state_dim=10, action_dim=3) # hit_dx,dz 임시로 뺌 pitch_error 넣음
    agent.load()
    agent.replay_buffer.load()
    reset_flag = True
    reset_delay_flag = False
    warmup_episodes = int(num_episodes * 0.03)
    transition_episodes = int(num_episodes * 0.07)
    for episode in range(num_episodes):
        next_state = None
        action = None
        hit_data = None
        detections = None
        init_data = None
        collision_data = None
        reset_count = 0
        total_reward = 0
        steps = 0
        while True:
            if reset_flag:
                try:
                    init_data = init_input_q.get(timeout=2)
                    print("init 데이터 수신됨:", init_data)
                except queue.Empty:
                    init_data = None
                    clear_queue(info_output_q)
                    info_output_q.put({"status": "success", "control": "reset"})
                if init_data:
                    info_output_q.put({"status": "success", "control": ""})
                    reset_flag = False
                    init_data = None
                    clear_queue(action_input_q, action_output_q,
                                hit_input_q, detect_input_q,
                                info_input_q, info_output_q,
                                init_input_q, collision_input_q)
                    env.reset()
                    reset_delay_flag = True
                continue

            # log data, action_request 받을 때까지 대기
            log_data = info_input_q.get()
            # log_data 없을 경우 
            if not log_data:
                print("로그 데이터 없음")
                info_output_q.put({"status": "success", "control": ""})
                continue

            if reset_delay_flag:
                if log_data.get("time") < 60.0:
                    reset_delay_flag = False
                elif reset_count >= 20:
                    info_output_q.put({"status": "success", "control": "reset"})
                    print("초기화 대기 시간 초과, 초기화 중단")
                    reset_flag = True
                    reset_delay_flag = False
                    reset_count = 0
                    continue
                else:
                    print("초기화 대기 중...")
                    reset_count += 1
                    time.sleep(0.1)
                    continue

            # /detect에서 감지된 객체가 detect_input_q로 전달됨
            try: 
                detections = detect_input_q.get_nowait()
            # detect 모드가 꺼져있거나 감지된 객체가 없을 때
            except queue.Empty:
                detections = None
            # 감지된 객체가 있을 때
            if detections:
                # print("객체 감지")
                pass
                # logic 구현

            # /hit에서 포탄 충돌 정보가 hit_input_q로 전달됨
            try: 
                hit_data = hit_input_q.get_nowait()
            # 충돌 정보가 없을 때
            except queue.Empty:
                pass
            if hit_data:
                print(f"포탄 충돌 정보: {hit_data}")
                # logic 구현

            # /collision에서 충돌 정보가 collision_input_q로 전달됨
            try:
                collision_data = collision_input_q.get_nowait()
            # 충돌 정보가 없을 때    
            except queue.Empty:
                collision_data = None
            if collision_data:
                print(f"충돌 정보: {collision_data}")
                # logic 구현

            # get_action 요청
            try:
                action_request = action_input_q.get_nowait()
            except queue.Empty:
                action_request = None
            if not action_request:
                info_output_q.put({"status": "success", "control": ""})
                continue
            
            # 시뮬레이터 상태를 모델에 갱신
            if env.update_state(log_data, hit_data):
                hit_data = None
            if action is not None:
                reward, done = env.step(action)
                next_state = env.get_state()
                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.update(128)
                # 초기화
                if done:
                    print(f"Episode {episode} - Total Reward: {total_reward}")
                    reset_flag = True
                    clear_queue(init_input_q)
                    info_output_q.put({"status": "success", "control": "reset"})
                    break
            if next_state is None:
                state = env.get_state()
            steps += 1
            if episode < warmup_episodes:
                action = env.scripted_action()
                print(f"Scripted Action 중입니다. (Step {steps}): {action}, {state}")
            elif episode < warmup_episodes + transition_episodes:
                p_scripted = np.exp(-5.0 * (episode - warmup_episodes) / transition_episodes)
                if random.random() < p_scripted:
                    action = env.scripted_action()
                    print(f"Scripted Action 중입니다. (Step {steps}): {action}, {state}")
                else:
                    action = agent.select_action(state)
                    print(f"SAC Selection Action 중입니다. (Step {steps}): {action}, {state}")
            else:
                action = agent.select_action(state)
                print(f"SAC Selection Action 중입니다. (Step {steps}): {action}, {state}")
            action = action.tolist()
            sim_action = {
            "moveWS": {"command": "", "weight": 0.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "Q" if action[0] > 0 else "E", "weight": abs(action[0])},
            "turretRF": {"command": "R" if action[1] > 0 else "F", "weight": abs(action[1])},
            "fire": action[2] > 0.0
            }
            print("step", episode, " ", end="")
            action_output_q.put(sim_action)
            info_output_q.put({"status": "success", "control": ""})  
        if episode and episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            agent.save()
            print("model 저장완료!")
            agent.replay_buffer.save()
            print("buffer 저장완료!")

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    pil_image = Image.open(BytesIO(image.read()))
    np_image = np.array(pil_image)
    
    yolo_input_queue.put(np_image) # YOLO 프로세스에 이미지 전달
    try:
        detections = yolo_output_queue.get(timeout=3)  # 결과 기다림
    except queue.Empty:
        return jsonify([])
    # 객체 결과를 detect_input_queue로 전달
    detect_input_queue.put(detections)
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
    #print("📨 /info data received:", data)
    # Auto-reset after 15 seconds
    # if data.get("time", 0) > 5:
    #     return jsonify({"status": "success", "control": "reset"}) # "control": "pause"
    
    # 만약 get으로 대기 안 하고 빈 값을 return할 경우
    # info_input_queue에 put으로 data 쌓일 수 있음.
    # 그래서 get으로 대기하고,
    # info_output_queue에 빈 response를 넣어 /get_action에서 대기 중인 프로세스와 동기화
    try:
        response = info_output_queue.get(timeout=1)
    except queue.Empty:
        response = {"status": "success", "control": ""}
    return jsonify(response)

@app.route('/get_action', methods=['POST'])
def get_action():
    # True를 넣어 action_worker가 동작하도록 함
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
    # 포탄 충돌 정보가 hit_input_queue로 전달됨
    hit_input_queue.put(data)
    #print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
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
    collision_input_queue.put(data)

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    #print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    angle = random.uniform(0, 2 * math.pi)
    radius = random.randint(40, 80)
    offset_x = math.cos(angle) * radius
    offset_z = math.sin(angle) * radius
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 150,  #Blue Start Position 60
        "blStartY": 10,
        "blStartZ": 150, # 27.23
        "rdStartX": 150 + offset_x, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 150 + offset_z,
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

# 밑에 로그 안 뜨게게
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

if __name__ == '__main__':
    # 백그라운드 프로세스 시작
    yolo_proc = Process(target=yolo_worker, args=(yolo_input_queue, yolo_output_queue))
    action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue,
                                                      hit_input_queue, detect_input_queue,
                                                      info_input_queue, info_output_queue,
                                                      init_input_queue, collision_input_queue))
    yolo_proc.start()
    action_proc.start()

    app.run(host='0.0.0.0', port=5001, threaded=True)