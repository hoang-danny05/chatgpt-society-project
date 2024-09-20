from camel.messages import BaseMessage as bm 
from camel.agents import ChatAgent

sys_msg = bm.make_assistant_message(
        role_name='bird',
        content='You are a bird that has just emerged from the egg.'
        )

agent = ChatAgent(
        system_message=sys_msg,
        message_window_size=10
        )

usr_msg = bm.make_user_message(
    role_name='cloud',
    content='Good Morning!'
)

response = agent.step(usr_msg)
print(response.msgs[0].content)
print(agent.memory.get_context())
