import random
from collections import defaultdict

import numpy as np

# from utils import vector_add, orientations, turn_right, turn_left
import operator

def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))

orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)

def turn_heading(heading, inc, headings=orientations):
    # print("AFTER TURN")
    # print(headings[(headings.index(heading) + inc) % len(headings)])
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    # print(heading)
    # print("RIGHT")
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    # print(heading)
    # print("LEFT")
    return turn_heading(heading, LEFT)


class MDP:

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""

        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None

    def check_consistency(self):

        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001


class MDP2(MDP):
    """
    Inherits from MDP. Handles terminal states, and transitions to and from terminal states better.
    """

    def __init__(self, init, actlist, terminals, transitions, reward=None, gamma=0.9):
        MDP.__init__(self, init, actlist, terminals, transitions, reward, gamma=gamma)

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return self.transitions[state][action]


class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=0.9):
        grid.reverse()  # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        # print(actlist)  
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
                # print(s, a)
                # print( transitions[s][a])
                # print("\n")
        # print(transitions)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions,
                     reward=reward, states=states, gamma=gamma)

    def calculate_T(self, state, action):
        if action:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]

    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0): 'R', (0, 1): 'U', (-1, 0): 'L', (0, -1): 'D', None: 'O'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


# ______________________________________________________________________________


""" [Figure 17.1]
A 4x3 grid environment that presents the agent with a sequential decision problem.
"""

sequential_decision_environment = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])


# ______________________________________________________________________________


def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration. [Figure 17.4]"""

    U1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max(sum(p * U[s1] for (p, s1) in T(s, a))
                                       for a in mdp.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))
        if delta <= epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. [Equation 17.4]"""

    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""

    return sum(p * U[s1] for (p, s1) in mdp.T(s, a))

# ______________________________________________________________________________

value_iteration(sequential_decision_environment, epsilon=0.001)
# for reward -2
sequential_decision_environment = GridMDP([[-2, -2, -2, +1],
                                           [-2, None, -2, -1],
                                           [-2, -2, -2, -2]],
                                          terminals=[(3, 2), (3, 1)])
value_iteration(sequential_decision_environment, epsilon=0.001)

#for reward 0.1
sequential_decision_environment = GridMDP([[0.1, 0.1, 0.1, +1],
                                           [0.1, None, 0.1, -1],
                                           [0.1, 0.1, 0.1, 0.1]],
                                          terminals=[(3, 2), (3, 1)])
value_iteration(sequential_decision_environment, epsilon=0.001)

#for reward 0.02
sequential_decision_environment = GridMDP([[0.02, 0.02, 0.02, +1],
                                           [0.02, None, 0.02, -1],
                                           [0.02, 0.02, 0.02, 0.02]],
                                          terminals=[(3, 2), (3, 1)])
value_iteration(sequential_decision_environment, epsilon=0.001)

#for reward 1
sequential_decision_environment = GridMDP([[1, 1, 1, +1],
                                           [1, None, 1, -1],
                                           [1, 1, 1, 1]],
                                          terminals=[(3, 2), (3, 1)])
value_iteration(sequential_decision_environment, epsilon=0.001)
