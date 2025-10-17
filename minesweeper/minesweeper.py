import itertools
import random

class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):
        
        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()


    def print(self):
        """
        Prints a text-based representation of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are within one row and column of a given cell, not including the cell itself.
        """
       # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count
    
    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells, and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count and self.count > 0:
            return set(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1
            print(f"Marked {cell} as a mine. Updated count: {self.count}. Remaining cells: {self.cells}")

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            print(f"Marked {cell} as safe. Remaining cells: {self.cells}")
        

class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):
        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def deduce_safes_from_marked_mines(self):
        """
        Deduces safe cells after marking some cells as mines.
        For any sentence where the number of mines equals the count,
        all other cells in that sentence must be safe.
        """
        new_safes = set()

        # Collect safe cells to mark after iteration
        for sentence in self.knowledge:
            if sentence.count == 0 and sentence.cells:
                new_safes.update(sentence.cells)

        # Mark safe cells after the loop to avoid set size change during iteration
        for safe in new_safes:
            if safe not in self.moves_made:
                self.mark_safe(safe)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighbouring cells have mines in them.

        This function should:
                1) mark the cell as a move that has been made
                2) mark the cell as safe
                3) add a new sentence to the AI's knowledge base
                based on the value of cell and count
                4) mark any additional cells as safe or as mines
                if it can be concluded based on the AI's knowledge base
                5) add any new sentences to the AI's knowledge base
                if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)
        self.mark_safe(cell)

        neighbours = set()
        i, j = cell
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if (x, y) == cell:
                    continue
                if 0 <= x < self.height and 0 <= y < self.width:
                    if (x, y) not in self.safes and (x, y) not in self.mines:
                        neighbours.add((x, y))

        if neighbours:
            new_sentence = Sentence(neighbours, count)
            self.knowledge.append(new_sentence)

        self.update_knowledge()
        self.infer_new_knowledge()

    def update_knowledge(self):
        """
        Updates the AI's knowledge by marking cells as safe or mines when they can be concluded based on the knowledge base.
        """
        knowledge_updated = True

        while knowledge_updated:
            knowledge_updated = False
            safe_cells = set()
            mine_cells = set()

            # Collect cells to be marked as safe or mines
            for sentence in self.knowledge:
                safes = sentence.known_safes()
                mines = sentence.known_mines()

                if safes:
                    safe_cells.update(safes)
                if mines:
                    mine_cells.update(mines)

            # Mark cells as safe
            for cell in safe_cells:
                if cell not in self.moves_made:
                    self.mark_safe(cell)
                    knowledge_updated = True

            # Mark cells as mines
            for cell in mine_cells:
                if cell not in self.moves_made:
                    self.mark_mine(cell)
                    knowledge_updated = True

            # Deduce additional safes from marked mines
            self.deduce_safes_from_marked_mines()

            # Remove empty sentences from knowledge base
            self.knowledge = [sentence for sentence in self.knowledge if sentence.cells]

        # Perform inference after updating knowledge
        self.infer_new_knowledge()


    def infer_new_knowledge(self):
        """
        Infers new knowledge by checking if any sentences in the knowledge base are subsets of each other and deriving new sentences accordingly.
        """
        new_sentences = []

        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                if sentence1 == sentence2:
                    continue
                if sentence1.cells.issubset(sentence2.cells):
                    inferred_cells = sentence2.cells - sentence1.cells
                    inferred_count = sentence2.count - sentence1.count
                    if inferred_count >= 0:  # Ensure count is not negative
                        if inferred_cells:
                            new_sentence = Sentence(inferred_cells, inferred_count)
                            if new_sentence not in self.knowledge and new_sentence not in new_sentences:
                                new_sentences.append(new_sentence)

        self.knowledge.extend(new_sentences)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move that has been made.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell
        return None

    def make_random_move(self):
        """
        Returns a random move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        possible_moves = []

        for i in range(self.height):
            for j in range(self.width):
                cell = (i, j)
                if cell not in self.mines and cell not in self.moves_made:
                    possible_moves.append(cell)

        if possible_moves:
            return random.choice(possible_moves)
        
        return None
