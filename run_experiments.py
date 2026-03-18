from src.parser import parse_path_output
from src.model import init_model, get_message, ask_model



def run(uses_tools, model, processor, img_path):
    # TODO: generate and add a random prompt
    user_prompt = (
        "Using the image, produce a complete action sequence for Spot to reach the other side "
        "of the wall. The direct path is blocked, so plan around the obstacle. "
    )

    messages = get_message(uses_tools, img_path, user_prompt)
    
    raw_output = ask_model(
        uses_tools=uses_tools,
        model=model,
        processor=processor,
        messages=messages
    )

    print("RAW OUTPUT:")
    print(raw_output)

    plan, error_msg = parse_path_output(raw_output)

    if error_msg is not None:
        print("ERROR:")
        print(error_msg)
        return

    print(plan)
    
    # get result
    # parse it
    # validate it
    return

def experiment():
    MODEL = "Qwen/Qwen3-VL-4B-Instruct"
    uses_tools = False

    model, processor = init_model(MODEL)

    IMAGE_PATH = "assets/wall_crossing_env.png"

    run(
        uses_tools=uses_tools,
        img_path=IMAGE_PATH,
        model=model, 
        processor=processor
    )



if __name__ == "__main__":
    experiment()