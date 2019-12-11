# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    "*** YOUR CODE HERE ***"
    'initialize'
    self.QV = util.Counter()

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    #QValue = self.QValues[state]
    if self.QV:
      return self.QV[(state,action)]
    else:
      return 0.0


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return 0.0
    else:
      #return max([self.getQValue(state, action) for action in self.getLegalActions(state)])
      QValue = []
      for action in legalActions:
        QValue.append(self.getQValue(state, action))
      Value = max(QValue)
      return Value

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return None
    else:
      #Value = self.getValue(state)
      QValue = []
      'best action can be more than one choice'
      paiAction = list()
      for action in legalActions:
        QValue.append(self.getQValue(state, action))
      Value = max(QValue)
      #paiAction = list()
      for action in legalActions:
        if self.getQValue(state, action) == Value:
          paiAction.append(action)
      return random.choice(paiAction)


  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    prob = self.epsilon
    if len(legalActions) == 0:
      return action
    if util.flipCoin(prob):
      action = random.choice(legalActions)
      return action
    else:
      action = self.getPolicy(state)
      return action


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    #QValue = self.QValues[state][action]
    QValue = self.getValue(state)
    nextQValue = self.getValue(nextState)
    alpha = float(self.alpha)
    gamma = float(self.discount)
    'formula'
    difference = reward + gamma * nextQValue - QValue
    update = QValue + alpha*difference
    self.QV[(state,action)] = update

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    "*** YOUR CODE HERE ***"
    self.w = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    featureVector = self.featExtractor.getFeatures(state, action)
    QVlist = list()
    #QValue = 0
    for feature in featureVector:
      weight = self.w[feature]
      newWeight = weight * featureVector[feature]
      QVlist.append(newWeight)
    QValue = sum(QVlist)
    #for QV in QVlist:
    #  QValue += QVlist[QV]
    return QValue



  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    featureVector = self.featExtractor.getFeatures(state, action)
    QValue = self.getQValue(state, action)
    nextQValue = self.getValue(nextState)
    alpha = self.alpha
    gamma = self.discount
    'difference formula'
    difference = reward + gamma*nextQValue - QValue
    'update formula'
    for feature in featureVector:
      weight = self.w[feature]
      update = weight + alpha*difference * featureVector[feature]
      self.w[feature] = update

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      pass