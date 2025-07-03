import queue

def clear_queue(*queues):
    for q in queues:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass