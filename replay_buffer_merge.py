import pickle
from collections import deque

def merge_replay_buffers(path1, path2, save_path, capacity=2000000): # capacity 2배배
    # 리플레이 버퍼 로드
    with open(path1, 'rb') as f:
        buffer1 = pickle.load(f)['buffer']
    with open(path2, 'rb') as f:
        buffer2 = pickle.load(f)['buffer']

    # 버퍼 병합
    merged = deque(maxlen=capacity)
    merged.extend(buffer1)
    merged.extend(buffer2)

    # 저장
    data = {'buffer': merged}
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Replay buffer 병합 완료: {save_path}")