from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType, RoleType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from datetime import datetime
import json
time_executed = datetime.now()

json_config = {}
with open("config.json", "r") as config_file:
    json_config = json.load(config_file)
# TODO: Add validation!!

# set the LLM model type and model config
model_platform = ModelPlatformType.OPENAI;
model_type = ModelType.GPT_3_5_TURBO;
# model_config = ChatGPTConfig(
#     temperature=1,  #sampling temp or randomness
#     n=3             #number of completion choices given
# )
model_config_dict = {
    "temperature":1,
    "n":1
}

#implies that you can have different LLMs talking to each other
model = ModelFactory.create(
    model_platform=model_platform,
    model_type=model_type,
    # model_config=model_config
    model_config_dict=model_config_dict
)


task_kwargs = {
    'task_prompt': json_config['task_prompt'],
    'with_task_specify': True,
    'task_specify_agent_kwargs': {'model': model}
}

user_role_kwargs = {
    'user_role_name': f"movie writer. ",
    'user_agent_kwargs': {
        'model': model
    }
}

assistant_role_kwargs = {
    'assistant_role_name': f"movie writer.",
    'assistant_agent_kwargs': {
        'model': model
    }
}

sys_msg_generator_kwargs = {
    "sys_msg_generator_kwargs": {
        "sys_prompts": {
            RoleType.USER:      f"{json_config['agent_one_prompt']}{json_config['user_prompt']}",
            RoleType.ASSISTANT: f"{json_config['agent_two_prompt']}{json_config['assistant_prompt']}" 
        },
        "sys_msg_meta_dict_keys" : {"user_role", "assistant_role", "task", "user_role_name", "assistant_role_name"},
    }
}

society = RolePlaying(
    **task_kwargs,             # The task arguments
    **user_role_kwargs,        # The instruction sender's arguments
    **assistant_role_kwargs,   # The instruction receiver's arguments
    **sys_msg_generator_kwargs,# The system message prompts
)

with open(f"output {json_config['task_name']}-{time_executed}.txt", "a") as outfile:
    outfile.write(f"user_name: {user_role_kwargs['user_role_name']}\n")
    outfile.write(f"user_prompt: {sys_msg_generator_kwargs['sys_msg_generator_kwargs']['sys_prompts'][RoleType.USER]}\n")
    outfile.write(f"assistant_name: {assistant_role_kwargs['assistant_role_name']}\n")
    outfile.write(f"assistant_prompt: {sys_msg_generator_kwargs['sys_msg_generator_kwargs']['sys_prompts'][RoleType.ASSISTANT]}\n")
    outfile.write("\n\n###START INTERACTIONS##\n\n")
    pass

# raise KeyboardInterrupt

def is_terminated(response):
    """
    Give alerts when the session shuold be terminated.
    """
    if response.terminated:
        role = response.msg.role_type.name
        reason = response.info['termination_reasons']
        print(f'AI {role} terminated due to {reason}')

    return response.terminated

def run(society, round_limit: int=10):
    # Get the initial message from the ai assistant to the ai user
    input_msg = society.init_chat()

    # Starting the interactive session
    for _ in range(round_limit):

        # Get the both responses for this round
        assistant_response, user_response = society.step(input_msg)

        # Check the termination condition
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # Get the results
        with open(f"output {json_config['task_name']}-{time_executed}.txt", "a") as outfile:
            user_str = f'[AI User] {user_response.msg.content}.\n'
            assi_str = f'[AI Assistant] {assistant_response.msg.content}.\n'
            outfile.write(user_str)
            outfile.write(assi_str)
            print(user_str)
            print(assi_str)

        # Check if the task is end
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break

        # Get the input message for the next round
        input_msg = assistant_response.msg

    return None

run(society)