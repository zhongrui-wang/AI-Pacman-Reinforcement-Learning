# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import random

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** MY CODE HERE ***"

        self.QValues = {}
        iterations = range(0,self.iterations)
        stateList = self.mdp.getStates()
        discount = self.discount

        for iteration in iterations:
            values = self.values.copy()
            for state in stateList:
                maxValue = None
                self.QValues[state] = {}
                for action in self.mdp.getPossibleActions(state):
                    value = 0
                    for next in self.mdp.getTransitionStatesAndProbs(state, action):
                        nextState = next[0]
                        prob = next[1]
                        stateReward = self.mdp.getReward(state, action, nextState)
                        stateValue = values[nextState]
                        'calculate values'
                        value += prob*(stateReward + discount*stateValue)
                    self.QValues[state][action] = value
                    if maxValue == None:
                        maxValue = value
                    else:
                        maxValue = max(value, maxValue)

                if not self.mdp.isTerminal(state):
                    self.values[state] = maxValue
                else:
                    self.values[state] = 0


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** MY CODE HERE ***"
        'Compute the Q-value'
        QValues = self.QValues[state][action]
        return QValues

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** MY CODE HERE ***"
        'initialize QValue and its corresponding action'
        possibleActions = self.mdp.getPossibleActions(state)
        'determine if state still has actions '
        if len(possibleActions) == 0:
            return None
        else:
            # Value = self.getValue(state)
            QValue = []
            'best action can be more than one choice'
            paiAction = list()
            'use for loop to compare QValue of each states'
            for action in possibleActions:
                QValue.append(self.getQValue(state, action))
            Value = max(QValue)
            # paiAction = list()
            for action in possibleActions:
                if self.getQValue(state, action) == Value:
                    paiAction.append(action)
            return random.choice(paiAction)

        #possibleActions = self.mdp.getPossibleActions(state)
        #maxQValue = 0
        #paiAction = None
        #'determine if state still has actions '
        #if not possibleActions:
        #    return None
        #'use for loop to compare QValue of each states'
        #for action in possibleActions:
        #    QValue = self.getQValue(state, action)
        #    if maxQValue < QValue:
        #        maxQValue = QValue
        #        paiAction = action
        #return paiAction

    #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)