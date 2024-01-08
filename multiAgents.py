# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="5", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            value = float('-inf')
            for action in gameState.getLegalActions(agent):
                value = max(value, self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            depth += 1
            value = float('inf')
            for action in gameState.getLegalActions(agent):
                value = min(value, self.alphabeta(agent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def heuristic(self, gameState):
        foods = gameState.getFood()
        (x, y) = gameState.getPacmanPosition()
        true_indices = []
        for i in range(20):
            for j in range(11):
                if foods[i][j]:
                    true_indices.append((i, j))
        closest_food = random.choice(true_indices)
        # min_distance = float("inf")
        # for i in range(20):
        #     for j in range(11):
        #         if foods[i][j] and (abs(i - x) + abs(j - y)) < min_distance:
        #             min_distance = abs(i - x) + abs(j - y)
        #             closest_food = (i, j)
        choosen_actions = []
        if x < closest_food[0]:
            choosen_actions.append('East')
        elif x > closest_food[0]:
            choosen_actions.append('West')
        if y < closest_food[1]:
            choosen_actions.append('North')
        elif y > closest_food[1]:
            choosen_actions.append('South')
        return choosen_actions
    

    def getAction(self, gameState: GameState):
        possible_actions = gameState.getLegalActions(0)
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), float('-inf'), float('inf')) for action in possible_actions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        if len(max_indices) > 1:
            suggested_actions = self.heuristic(gameState)
            performed_actions = []
            for action in max_indices:
                if possible_actions[action] in suggested_actions:
                    performed_actions.append(possible_actions[action])
            if len(performed_actions) > 0:
                return(random.choice(performed_actions))
        chosen_index = random.choice(max_indices)
        return possible_actions[chosen_index]
    