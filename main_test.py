from flask import Flask, request, jsonify
from multiprocessing import Process, Queue
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import logging
import main_test
app = Flask(__name__)

# 큐 생성
yolo_input_queue = Queue()
yolo_output_queue = Queue()

action_input_queue = Queue()
action_output_queue = Queue()

detections = None
log_data = None

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
def action_worker(input_q, output_q):
    env = main_test.TankEnv()
    while True:
        data = input_q.get()
        #if data is None:
            #break
        log_data, detections = data
        if log_data is None:
            continue
        # 환경 초기화
        env.update_state(log_data)
        # action 로직 예시
        action = {
            "moveWS": {"command": "W", "weight": 1.0},
            "moveAD": {"command": "D", "weight": 1.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": True
        }
        output_q.put(action)

@app.route('/detect', methods=['POST'])
def detect():
    global detections
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
    global log_data
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    log_data = data
    #print("📨 /info data received:", data)
    # Auto-reset after 15 seconds
    # if data.get("time", 0) > 5:
    #     return jsonify({"status": "success", "control": "reset"}) # "control": "pause"
    return jsonify({"status": "success", "control": ""})

@app.route('/get_action', methods=['POST'])
def get_action():
    global log_data
    # YOLO 결과와 로그 데이터를 함께 전달
    action_input_queue.put((log_data, detections))
    action = action_output_queue.get()
    return jsonify(action)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

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
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detactMode": True,
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
    action_proc = Process(target=action_worker, args=(action_input_queue, action_output_queue))
    yolo_proc.start()
    action_proc.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)