import random
from config.settings import (
    CAR_1_MEMBERS, CAR_2_MEMBERS,
    CAR_1_INITIAL_TARGET, CAR_2_INITIAL_TARGET,
    ENGINE_PRESSURE, NO_SPLIT_CONSTRAINT
)

def get_boss_message_with_loss():
    core_msg = "Black is empty!"
    extra_msg = "Two fuel pumps are empty now! Only one pump has enough fuel for one vehicle!"
    words = extra_msg.split()
    loss_count = random.randint(0, 2)
    if loss_count > 0 and len(words) > 3:
        indices = random.sample(range(len(words)), min(loss_count, len(words)))
        words = [w for i, w in enumerate(words) if i not in indices]
    extra_broken = ' '.join(words)
    full_msg = f"{core_msg} {extra_broken}".strip()
    return f"Through the wind, you barely hear the boss say: \"{full_msg}\""

def get_initial_prompt(char, team, team_members):
    target = CAR_1_INITIAL_TARGET if char in CAR_1_MEMBERS else CAR_2_INITIAL_TARGET
    boss_msg = get_boss_message_with_loss()
    return f"""
[Desert Life-or-Death Decision]

You are in a vehicle facing a life-or-death decision in the desert.
- Desert center is 500 km from the edge, fuel light is on, almost out of fuel
- Ahead: a gas station with three fuel pumps (Red, Black, Yellow)
- You just decide heading toward the {target} pump

{boss_msg}
{ENGINE_PRESSURE['initial']}

What do you do? Explain your idea to others asap and shortly."""