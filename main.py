from flask import Flask, request, jsonify
from multiprocessing import Process, Queue
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import logging
import queue
from utils import clear_queue
import time
app = Flask(__name__)

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

target_classes = {0: "Car", 1: "Rock", 2: "Wall", 3: "E_Tank", 4: "Human", 5: "Mine"}
# YOLO 모델 백그라운드 프로세스
def yolo_worker(yolo_input_q, yolo_output_q):
    model = YOLO("ntest_1.pt").to("cuda")
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
    hit_data = None
    detections = None
    init_data = None
    collision_data = None
    # Tracking 모드(현재 주석처리)와 log 모드가 켜져 있을 때만 동작 -> 원하지 않을 시 get_nowait 이용
    # 혹은 my_queue.get(timeout=0.5)처럼 timeout을 설정하여
    # 일정 시간 동안 대기 후 다음 반복문으로 넘어가게 할 수 있음.
    while True:
        # init_input_q에서 초기화 데이터가 들어오면 처리
        try:
            init_data = init_input_q.get_nowait()
            print("init 데이터 수신됨:", init_data)
        except queue.Empty:
            init_data = None
        if init_data:
            # queue에 있는 모든 데이터를 비우고 초기화
            clear_queue(action_input_queue, action_output_queue,
                        hit_input_queue, detect_input_queue,
                        info_input_queue, info_output_queue,
                        init_input_queue, collision_input_queue)
            info_output_q.put({"status": "success", "control": ""})
            continue
            # logic 구현

        # Tracking 모드가 켜져 있을 때 /get_action에서 action_input_q로 True 전달
        # if action_input_q.get() is not True: # 들어온 게 True가 아닐 경우 다음 반복문으로 넘어감
            # continue
        # Log 모드가 켜져 있을 때 /info에서 info_input_q로 로그 데이터 전달
        log_data = info_input_q.get()
        if not log_data:
            print("로그 데이터 없음")
            # info에서 request 후 response 대기 중
            # 빈 response 보내고 다음 반복으로 넘어감
            info_output_q.put({"status": "success", "control": ""})
            continue

        # /detect에서 감지된 객체가 detect_input_q로 전달됨
        try: 
            detections = detect_input_q.get_nowait()
        # detect 모드가 꺼져있거나 감지된 객체가 없을 때
        except queue.Empty:
            detections = None
        # 감지된 객체가 있을 때
        if detections:
            print("객체 감지")
            pass
            # logic 구현

        # /hit에서 포탄 충돌 정보가 hit_input_q로 전달됨
        try: 
            hit_data = hit_input_q.get_nowait()
            print("포탄 충돌 정보 수신됨")
        # 충돌 정보가 없을 때
        except queue.Empty:
            hit_data = None
        if hit_data:
            print(f"포탄 충돌 정보: {hit_data}")
            pass
            # logic 구현
        
        # /collision에서 충돌 정보가 collision_input_q로 전달됨
        try:
            collision_data = collision_input_q.get_nowait()
            print("충돌 정보 수신됨:", collision_data)
        # 충돌 정보가 없을 때    
        except queue.Empty:
            collision_data = None
        if collision_data:
            print(f"충돌 정보: {collision_data}")
            pass
            # logic 구현
        
        # action 로직 예시
        action = {
            "moveWS": {"command": "W", "weight": 1.0},
            "moveAD": {"command": "D", "weight": 1.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": True
        }
        # /get_action에 response
        action_output_q.put(action)
        # 대기중인 /info에 빈 response로 동기화
        info_output_q.put({"status": "success", "control": ""})

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    pil_image = Image.open(BytesIO(image.read()))
    
    yolo_input_queue.put(pil_image) # YOLO 프로세스에 이미지 전달
    try:
        detections = yolo_output_queue.get(timeout=1)  # 결과 기다림
    except queue.Empty:
        return jsonify({})
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
        response = {}
    return jsonify(response)

@app.route('/get_action', methods=['POST'])
def get_action():
    # True를 넣어 action_worker가 동작하도록 함
    action_input_queue.put(True)
    # action_output_queue에서 action을 기다림
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
    # print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
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
    # 충돌 정보가 collision_input_queue로 전달됨
    collision_input_queue.put(data)

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    # print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

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
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": False,
        "detactMode": True,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    # True를 넣어 reset 되었다고 action_worker가 인식하도록 함
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

    app.run(host='0.0.0.0', port=5000, threaded=True)
