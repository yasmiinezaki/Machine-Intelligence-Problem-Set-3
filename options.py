# This file contains the options that you should modify to solve Question 2

# applied a general rule that the farther we want a destination the bigger noise value
# to make the agent go for the closer reward apply a heavier discount and increase living penalty
# to make an agent choose any terminating qcondition increase living penalty by a value greater than any terminal value
# to stay in the game forever increase living reward above or equal any terminal values
def question2_1(): #done
    return {
        "noise": 0.2,
        "discount_factor": 0.8,
        "living_reward": -3
    }

def question2_2(): #done
    return {
        "noise": 0.05,
        "discount_factor": 0.1,
        "living_reward": -0.5
    }

def question2_3(): #done
    return {
        "noise": 0.01,
        "discount_factor": 1.0,
        "living_reward": -2.5
    }

def question2_4():
    return {
        "noise": 0.2,
        "discount_factor": 1.0,
        "living_reward": -0.3
    }

def question2_5(): #done
    return {
        "noise": 0.0,
        "discount_factor": 0.5,
        "living_reward": 10.0
    }

def question2_6(): #done
    return {
        "noise": 0.0,
        "discount_factor": 0.5,
        "living_reward": -20
    }