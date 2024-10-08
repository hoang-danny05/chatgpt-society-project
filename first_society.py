from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig

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
    'task_prompt': 'You are acting a scene for a movie. You both meet outside a rally. ',
    'with_task_specify': True,
    'task_specify_agent_kwargs': {'model': model}
}

user_role_kwargs = {
    'user_role_name': 'You are roleplaying as an actor for a movie scene. Your name is Jose and you are a Mexican-American male in your 20s. Your family has experienced a lot of hardship due to border policy. Your dad was deported when you were eight years old. This is a traumatic experience that you still deal with today. You are protesting a rally that you are against. You are passionate about the topic of border policy and are well-informed. You will stand your ground on your beliefs. You are normally composed, unless the right person provokes you. ',
    'user_agent_kwargs': {'model': model}
}

assistant_role_kwargs = {
    'assistant_role_name': 'You are roleplaying as an actor for a movie scene. Your name is Connor and you are a Caucasian male in your 60s, who hate current border policy and is paranoid about \"migrants\". You refer to immigrants as \"ailens\" and anytime the topic comes up about border policy, you get super defensive and rude, and stand your ground on your beliefs. You sometimes get heated enough to swear. ',
    'assistant_agent_kwargs': {'model': model}
}

society = RolePlaying(
    **task_kwargs,             # The task arguments
    **user_role_kwargs,        # The instruction sender's arguments
    **assistant_role_kwargs,   # The instruction receiver's arguments
)

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
        with open("output.txt", "a") as outfile:
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