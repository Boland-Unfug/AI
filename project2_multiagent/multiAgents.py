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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        #print("########## NEXT STATE #############")
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"
        #print("Direction Chosen:", legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (pos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        #print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newpos = successorGameState.getPacmanPosition()
        pos = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #print("Testing ", action)
        # very simple situation:
        # while food not found, search one firther until you hit a wall
        # if there is a ghost, run away (- score)
        
        score = 0

        if (currentGameState.hasFood(newpos[0],newpos[1]) == True):
            score += 1
        
        if (successorGameState.hasFood(newpos[0],newpos[1])==True):
            score += 1

        foodFound = False
        stop = False
        x = pos[0]
        y = pos[1]
        while (currentGameState.hasWall(x,y) == False and foodFound == False and action != "Stop"):
            
            if (currentGameState.hasFood(x,y) == True):
                foodFound = True
                distance = abs(x-pos[0])+ abs(y-pos[1])
                if (distance == 0):
                    score += 2
                else:
                    score += 1/distance

            if(action == "East"):
                x += 1
            if(action == "West"):
                x -= 1
            if(action == "North"):
                y += 1
            if(action == "South"):
                y -= 1

        if (action == "Stop"):
            score -=2
                    
        for ghosts in currentGameState.getGhostPositions():
            if(pos == ghosts or newpos == ghosts): # why is this not detecting the second ghost?
                score -= 10

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        score, action = MinimaxAgent.miniMaxi(gameState,self.depth)

        return action


    def miniMaxi(gameState:GameState,depth, agent=-1):
        #print("JUST BC")
        #start at agent -1 because I need to update before recursing, and need to start at 0
        bestScore = -9999
        bestMove = ""
        if (gameState.isLose() == True or gameState.isWin() == True): # why can't I put this with the depth check
                return gameState.getScore(), bestMove
        
        # updates agents and depth if at last agent
        if(agent < gameState.getNumAgents()-1): agent += 1
        else:
            agent = 0
            depth -= 1

        if depth > 0:
            #gets best move for the agent and returns results
            if agent == 0:
                bestScore, bestMove = MinimaxAgent.maxi(gameState,depth,agent)
            else:
                bestScore, bestMove = MinimaxAgent.mini(gameState,depth,agent)

            return bestScore, bestMove
        # returns max depth state score, and its best move, ""
        return gameState.getScore(), bestMove
        


    def maxi (gameState:GameState,depth, agent):
        score = -9999
        move = ""
        for action in gameState.getLegalActions(0):
            new = MinimaxAgent.miniMaxi(gameState.generateSuccessor(0,action),depth,agent)
            newScore = new[0]
            if score < newScore:
                score = newScore
                move = action
            
        return score, move
    
    def mini (gameState:GameState,depth, agent):
        score = 9999
        move = ""
        for action in gameState.getLegalActions(agent):
                new = MinimaxAgent.miniMaxi(gameState.generateSuccessor(agent,action),depth,agent)
                newScore = new[0]
                if score > newScore:
                    score = newScore
                    move = action
        return score, move
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action, alpha, beta = AlphaBetaAgent.miniMaxi(gameState,self.depth)

        return action

    
    def miniMaxi(gameState:GameState,depth, agent=-1, alpha = -9999, beta = 9999):
        #print("JUST BC")
        #start at agent -1 because I need to update before recursing, and need to start at 0
        bestScore = -9999
        bestMove = ""
        if (gameState.isLose() == True or gameState.isWin() == True): # why can't I put this with the depth check
                return gameState.getScore(), bestMove, alpha, beta
        
        # updates agents and depth if at last agent
        if(agent < gameState.getNumAgents()-1): agent += 1
        else:
            agent = 0
            depth -= 1

        if depth > 0:
            #gets best move for the agent and returns results
            if agent == 0:
                bestScore, bestMove, alpha, beta = AlphaBetaAgent.maxi(gameState,depth,agent,alpha,beta)
            else:
                bestScore, bestMove, alpha, beta = AlphaBetaAgent.mini(gameState,depth,agent,alpha,beta)

            return bestScore, bestMove, alpha, beta
        # returns max depth state score, and its best move, ""
        return gameState.getScore(), bestMove, alpha, beta
        


    def maxi (gameState:GameState,depth, agent, alpha, beta):
        score = -9999
        move = ""
        for action in gameState.getLegalActions(0):
            new = AlphaBetaAgent.miniMaxi(gameState.generateSuccessor(0,action),depth,agent, alpha, beta)
            newScore = new[0]
            if score < newScore:
                score = newScore
                move = action
            
            if score > beta:
                return score, move, alpha, beta
            if alpha < score: alpha = score

        return score, move, alpha, beta
    
    def mini (gameState:GameState,depth, agent, alpha, beta):
        score = 9999
        move = ""
        for action in gameState.getLegalActions(agent):
                new = AlphaBetaAgent.miniMaxi(gameState.generateSuccessor(agent,action),depth,agent, alpha, beta)
                newScore = new[0]
                if score > newScore:
                    score = newScore
                    move = action
                if score < alpha:
                    return score, move, alpha, beta
                if beta > score: beta = score
        return score, move, alpha, beta
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        score, action = ExpectimaxAgent.miniMaxi(gameState,self.depth)
        #print("score", score), print("action", action)
        return action


    def miniMaxi(gameState:GameState,depth, agent=-1):
        #print("JUST BC")
        #start at agent -1 because I need to update before recursing, and need to start at 0
        bestScore = -9999
        bestMove = ""
        if (gameState.isLose() == True or gameState.isWin() == True): # why can't I put this with the depth check
                return gameState.getScore(), bestMove
        # updates agents and depth if at last agent
        #print("NUM AGENTS", gameState.getNumAgents())
        if(agent < gameState.getNumAgents()-1):
            agent += 1
            #print("agent", agent, "depth", depth, "legal actions", "bestMove", bestMove, "bestScore", bestScore)
        else:
            agent = 0
            depth -= 1
            #print("bestscore", bestScore, "At Depth", depth, "Agent", agent, "Action", bestMove)
        # I will represent chance nodes as odd
        if depth > 0:
            #gets best move for the agent and returns results
            #print("agent",agent)
            if agent == 0:
                #print("maxi", agent)
                bestScore, bestMove = ExpectimaxAgent.maxi(gameState,depth,agent)
            else:
                #print("mini", agent)
                bestScore, bestMove = ExpectimaxAgent.mini(gameState,depth,agent)

            return bestScore, bestMove
        # returns max depth state score, and its best move, ""
        return gameState.getScore(), bestMove
        

    def maxi (gameState:GameState,depth, agent):
        #print("maxi", agent, "depth", depth)
        score = -9999
        move = ""
        #print("maxi moves", gameState.getLegalActions(agent))
        for action in gameState.getLegalActions(agent):
            new = ExpectimaxAgent.chance(gameState,depth,agent,action)
            #print("new", new)
            newScore = new[0]
            if score < newScore:
                score = newScore
                move = action
            
        return score, move
    
    def mini (gameState:GameState,depth, agent):
        #print("mini", agent, "depth", depth)
        score = 9999
        move = ""
        for action in gameState.getLegalActions(agent):
                new = ExpectimaxAgent.chance(gameState,depth,agent,action)
                #print("new 2", new)
                newScore = new[0]
                if score > newScore:
                    score = newScore
                    move = action
        return score, move

    def chance (gameState:GameState,depth, agent, move):
        score = 0
        new = ExpectimaxAgent.miniMaxi(gameState.generateSuccessor(agent,move),depth,agent)
        newScore = new[0]
        score += newScore
        score = score / len(gameState.getLegalActions(agent))
        return score, move

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    print("is this even triggering")
    score = 0

    if currentGameState.isWin(): return 100
    elif currentGameState.isLose(): return -100

    for action in currentGameState.getLegalActions(0):
        nextState = currentGameState.generatePacmanSuccessor(action)
        if nextState.isWin(): return 100
        elif nextState.isLose(): return -100
        elif nextState.hasFood(): score += 1
    #this should evaluate the current state of the game, and return a score
    #the score should be a number, and the higher the number the better the state
    # I should be able to call it to an arbitrary depth, and it should return a score
    
        

    return score

# Abbreviation
better = betterEvaluationFunction
