 
import ray
from actors.prefiller import PrefillActor
from actors.decoder import DecodingActor

if __name__ == "__main__":
    decoder = DecodingActor.remote()
    prefiller = PrefillActor.remote(decoder)

    user_requests = ["Request 1", "Request 2", "Request 3"]

    tasks = []
    for request in user_requests:
        task = prefiller.handle_request.remote(request)
        tasks.append(task)

    ray.get(tasks)

    import time
    time.sleep(1)

    ray.shutdown()