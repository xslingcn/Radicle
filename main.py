import ray
import uuid

from config.actors.controller_config import ControllerConfig
from config.actors.decoder_config import DecoderConfig 
from config.actors.prefiller_config import PrefillerConfig

from actors.controller import Controller
from actors.decoder import DecodingActor
from actors.prefiller import PrefillActor
from data.role import Role

if __name__ == "__main__":
    ray.init()
    
    try:
        controller = Controller.remote(ControllerConfig())
        ray.get(controller.start.remote())
        
        for _ in range(2):
            role = ray.get(controller.request_role.remote())
            if role == Role.PREFILLER:
                print("Role: Prefiller")
                prefiller = PrefillActor.remote(
                    PrefillerConfig(),
                    "mock_model", 
                    "mock_tokenizer",
                    controller
                )
                print(prefiller)
                ray.get(prefiller.set_ref.remote(prefiller))
                ray.get(prefiller.start.remote())
                ray.get(controller.register.remote(prefiller, Role.PREFILLER))
            else:
                print("Role: Decoder")
                decoder = DecodingActor.remote(
                    DecoderConfig(),
                    "mock_model",
                    "mock_tokenizer",
                    controller
                )
                ray.get(decoder.set_ref.remote(decoder))
                ray.get(decoder.start.remote())
                ray.get(controller.register.remote(decoder, Role.DECODER))
        
        for _ in range(10):
            ray.get(controller.add_request.remote(str(uuid.uuid4()), "Hello, world!"))
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        ray.shutdown()