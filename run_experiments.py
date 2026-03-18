from src.simulator import simulate_plan
from src.parser import parse_path_output
from src.model import init_model, get_message, ask_model

IMAGE_PATH = "assets/wall_crossing_env.png"



def run(uses_tools, model, processor, img_path) -> tuple[bool, bool]:
    structure = False
    completion = False

    # Prompt Model
    # TODO: generate and add a random prompt
    # user_prompt = (
    #     "Using the image, produce a complete action sequence for Spot to reach the other side "
    #     "of the wall. The direct path is blocked, so plan around the obstacle. "
    # )
    user_prompt = (
        "Spot is at (0,0), heading 0 means facing +x 'up'. "
        "There is a wall right in front of spot with the bounds x in [4.5, 5.5], y in [-2, 2]. "
        "Produce a complete action sequence for Spot to reach the other side of the wall. "
        "The direct path is blocked, so plan around the obstacle. "
        "The target bounds are x in [6, 10], y in [-3, 3]."
    )

    messages = get_message(uses_tools, img_path, user_prompt)
    
    raw_output = ask_model(
        uses_tools=uses_tools,
        model=model,
        processor=processor,
        messages=messages
    )

    # Validate Response
    print("RAW OUTPUT:")
    print(raw_output)

    plan, error_msg = parse_path_output(raw_output)
    
    if error_msg is not None:
        print("ERROR:")
        print(error_msg)
        return structure, completion
    
    structure = True

    # Assess task success
    plan_results = simulate_plan(plan)
    spot_success = plan_results["success"]
    final_spot = plan_results["final_spot"]

    if plan_results["collided"]:
        print(f"Spot collided at: {final_spot}")
    elif spot_success:
        print(f"Spot successful, reached: {final_spot}")
        completion = True
    elif not spot_success:
        print(f"Spot did not reach target. Final position: {final_spot}")

    return structure, completion

def experiment():
    MODEL = "Qwen/Qwen3-VL-4B-Instruct"
    uses_tools = False
    uses_image = False

    model, processor = init_model(MODEL)

    if uses_image:
        img_path = IMAGE_PATH
    else:
        img_path = None
    
    structure, completion = run(
        uses_tools=uses_tools,
        img_path=img_path,
        model=model, 
        processor=processor
    )



if __name__ == "__main__":
    experiment()