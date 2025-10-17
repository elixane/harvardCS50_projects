"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count the number of Xs and Os on the board
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    
    # If X has the same number of moves as O, it's X's turn
    if x_count == o_count:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Create empty set of all possible actions
    board_actions = set()

    # For each cell check whether an action is available, and if so, add it to the list of actions
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == EMPTY:
                board_actions.add((i, j))

    return board_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    
    # Check if the action is valid
    if board[i][j] is not EMPTY:
        raise Exception("Invalid action: cell is not empty.")
    
    # Create a deep copy of the board
    new_board = copy.deepcopy(board)
    
    # Determine the current player and apply the action
    current_player = player(board)
    new_board[i][j] = current_player
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check if there is a winner in the rows 
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not EMPTY:
            return row[0]  # Return the winner ('X' or 'O')
        
    # Chheck if there is a winner in the columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not EMPTY:
            return board[0][col]

    # Check if there is a winner in the diagonalss
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]
    
    # Return nothing if there is no winner yet
    return None 


def terminal(board):
    """
    Returns True if the game is over, False otherwise.
    """
    # Check if there is a winner
    if winner(board) is not None:
        return True
    
    # Check if there are no moves left (all cells are filled)
    if all(cell is not EMPTY for row in board for cell in row):
        return True
    
    # Game is not over
    return False           


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == "X":
        return 1
    elif winner(board) == "O":
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal move for the current player on the given board.
    """
    # Check if the board is in a terminal state
    if terminal(board):
        return None

    # Get the current player
    current_player = player(board)

    # Initialize the best move and the best value
    if current_player == X:
        best_value = -math.inf
        best_move = None
        
        # Explore all possible actions
        for action in actions(board):

            # Simulate the board after making the move
            new_board = result(board, action)
            
            # Recursively call minimax for the opponent
            value = minimax_value(new_board)
            
            # Update the best move if needed
            if value > best_value:
                best_value = value
                best_move = action
        
    else:  # current_player == O
        best_value = math.inf
        best_move = None
        
        # Explore all possible actions
        for action in actions(board):
            
            # Simulate the board after making the move
            new_board = result(board, action)
            
            # Recursively call minimax for the opponent
            value = minimax_value(new_board)
            
            # Update the best move if needed
            if value < best_value:
                best_value = value
                best_move = action

    return best_move


def minimax_value(board):
    """
    Returns the utility value of the board for the current player using the minimax algorithm.
    """
    if terminal(board):
        return utility(board)
    
    current_player = player(board)
    
    if current_player == X:
        best_value = -math.inf
        for action in actions(board):
            new_board = result(board, action)
            value = minimax_value(new_board)
            best_value = max(best_value, value)
        return best_value
    else:  # current_player == O
        best_value = math.inf
        for action in actions(board):
            new_board = result(board, action)
            value = minimax_value(new_board)
            best_value = min(best_value, value)
        return best_value

