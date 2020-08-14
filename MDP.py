"""
MDP solved by Value Iteration

Using the Bellman equations

V(s) = min_action sum_s' T(s, a, s')[C(s, a, s') + V(s')]

"""
import numpy

# one "state factor", or one "state variable"
# this is location with 6 values
locations = ['a', 'b', 'c', 'd', 'e', 'f']
# we can directly map these to states in the MDP
# this may not always be the case
states = locations

# some actions are only enabled in some states
# e.g. action a_b is only enabled in state a
# action definition is a mapping from an action name, e.g. 'a_b'
# to enabled states, as a set, and outcome states
actions = {}

actions['a_b'] = [
    # enabled states
    {'a'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['b', 0.7, 2], ['a', 0.3, 2]]
]

actions['b_c'] = [
    # enabled states
    {'b'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['c', 1.0, 1]]
]

actions['b_e'] = [
    # enabled states
    {'b'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['e', 1.0, 1]]
]

actions['d_a'] = [
    # enabled states
    {'d'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['a', 0.9, 2], ['d', 0.1, 2]]
]

actions['d_e'] = [
    # enabled states
    {'d'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['e', 0.7, 2], ['d', 0.3, 2]]
]


actions['e_b'] = [
    # enabled states
    {'e'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['b', 1.0, 1]]
]

actions['e_f'] = [
    # enabled states
    {'e'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['f', 1.0, 1]]
]

actions['f_e'] = [
    # enabled states
    {'f'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['e', 1.0, 1]]
]

actions['f_c'] = [
    # enabled states
    {'f'}, 
    # list of probabilistic outcomes, i.e. the states we reach via this action
    [['c', 0.9, 2], ['f', 0.1, 2]]
]


def print_enable_states():
    # for each state, print out enabled action
    for state in states:
        for action, action_model in actions.items(): 
            if state in action_model[0]:
                print(action, 'enabled in', state)

def value_iteration(goal_states,epsilon):
    # value iteration for SSPs
    #goal_states = {'c'}
    # initialise values for iteration 0
    # trying to minimise cost, so initialise to a high cost
    init_value = 100
    global values
    values = {}
    # initialise value for all states
    for state in states:
        if state in goal_states:
            values[state] = 0
        else:    
            values[state] = init_value

    max_residual = init_value

    while max_residual > epsilon:
    # q(s, a) - cost of action a in state s
    # v(s) is min_a of q(s, a)
        global q_values
        q_values = {}
        # for each state
        for state in states:
            if state not in goal_states:
                # initialise q_values for state to empty dictionary
                q_values[state] = {}
                # find each action i can apply in state
                for action, action_model in actions.items(): 
                    # access the action model
                    states_where_action_is_enabled = action_model[0]
                    probabilistic_outcomes = action_model[1]

                    if state in states_where_action_is_enabled:
                        # for each enabled action, sum over outcomes
                        q_value = 0
                        for outcome in probabilistic_outcomes:
                            outcome_state = outcome[0]
                            outcome_probability = outcome[1]
                            outcome_cost = outcome[2]

                            # accumulate q_value per successor state
                            q_value += outcome_probability * (outcome_cost + values[outcome_state])

                        # store the q value
                        q_values[state][action] = q_value
                        # print('q_value for', state, action, q_values[state][action])

            # calculate value per state
        iteration_residual = 0
        for state in states:
            min_q = init_value * 2

            if state not in goal_states:
                # look at all the q values for each state
                for action, q_value in q_values[state].items():

                    # keep track of the minimum q value
                    # TODO: if we're building a policy, keep track of actions here
                    if q_value < min_q:
                        min_q = q_value
                
                residual = abs(values[state] - min_q)
                values[state] = min_q
                
                if residual > iteration_residual:
                    iteration_residual = residual

        #print(values)
        #print(iteration_residual)
        max_residual = iteration_residual

def policy(initial_state,goal_states): 
    i_state = initial_state  
    for state in states: 
        if state not in goal_states:
            for action, q_value in q_values[state].items():
                q_values[state][action] = q_value
                #print(state, action, q_values[state][action])
    print('The optimal policy is:', initial_state,'to',end='')
    while (1):
        next_action = (min(q_values[i_state],key=lambda k:q_values[i_state][k]))
        i_state =  next_action[2]
        if i_state == goal_states:
            print('',i_state,'end')
            break
        print('',i_state,'to',end='')
        #print(i_state,'-')
if __name__ == "__main__":
    #print_enable_states()
    value_iteration('c',0.00001)
    policy('d','c')
