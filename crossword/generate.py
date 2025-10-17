import sys
from collections import deque

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
        # Make a copy of the domain to iterate over
            for word in self.domains[v].copy():
                # If the word's length doesn't match the variable's length, remove it
                if len(word) != v.length:
                    self.domains[v].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revision = False
        overlap = self.crossword.overlaps[x, y]

        # If there's no overlap between x and y, no revision is necessary
        if overlap is None:
            return revision

        i, j = overlap  # The indices where x and y overlap

        # Make a list of values to remove from x's domain
        to_remove = set()

        for word_x in self.domains[x]:
            # Check if there is some word in y's domain that is consistent
            is_consistent = False
            for word_y in self.domains[y]:
                if word_x[i] == word_y[j]:  # Characters at overlap should match
                    is_consistent = True
                    break
            
            # If no word in y's domain is consistent with word_x, mark for removal
            if not is_consistent:
                to_remove.add(word_x)

        # Remove the inconsistent values from x's domain
        for word in to_remove:
            self.domains[x].remove(word)
            revision = True

        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        #Create a list of all arcs in the problem
        if arcs is None:
            arcs = [(x, y) for x in self.crossword.variables for y in self.crossword.neighbors(x)]

        #Initialise queue of all arcs in the list
        queue = deque(arcs)

         # Process the queue until empty
        while queue:
            x, y = queue.popleft()

            # Revise x to make it arc-consistent with y
            if self.revise(x, y):
                # If x's domain is empty, no solution is possible
                if len(self.domains[x]) == 0:
                    return False

                # Add all arcs (z, x) back to the queue (except (y, x) to avoid loops)
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))

        # If arc consistency is enforced successfully, return True
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # Check if every variable in the crossword has a corresponding value in the assignment
        for var in self.crossword.variables:
            # If a variable is missing from the assignment or its value is None, return False
            if var not in assignment or assignment[var] is None:
                return False

        # If every variable has been assigned a value, return True
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
         # Ensure all values are distinct (no repeated words in the assignment)
        words = list(assignment.values())
        if len(words) != len(set(words)):
            return False

        # Ensure every word has the correct length
        for var, word in assignment.items():
            if len(word) != var.length:
                return False

        # Ensure there are no conflicts between neighboring variables
        for var in assignment:
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap is not None:
                        i, j = overlap
                        # Check if the characters in the overlapping positions are the same
                        if assignment[var][i] != assignment[neighbor][j]:
                            return False
            
        # If all conditions are satisfied, return True
        return True
    
    
    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        def conflicts(value):
            """
            Calculate the number of values eliminated from the domains of neighboring unassigned variables
            if `var` is assigned to `value`.
            """
            conflict_count = 0
            for neighbor in self.crossword.neighbors(var):
                # Skip neighbors already assigned
                if neighbor in assignment:
                    continue
                
                # Get overlap positions between var and its neighbor
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is None:
                    continue
                
                i, j = overlap
                # Count how many values in neighbor's domain conflict with value
                for neighbor_value in self.domains[neighbor]:
                    if value[i] != neighbor_value[j]:
                        conflict_count += 1

            return conflict_count

        # Get domain values sorted by the least-constraining values heuristic (fewest conflicts first)
        return sorted(self.domains[var], key=conflicts)


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Filter unassigned variables
        unassigned_vars = [v for v in self.crossword.variables if v not in assignment]
        
        # Apply MRV heuristic: sort by the number of remaining values (domain size)
        def mrv(var):
            return len(self.domains[var])

        # Apply Degree heuristic: sort by the number of neighbors (in descending order)
        def degree(var):
            return len(self.crossword.neighbors(var))

        # Sort variables by MRV first, and then by degree in case of ties
        return min(unassigned_vars, key=lambda var: (mrv(var), -degree(var)))


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        # Base case: If assignment is complete, return the assignment
        if self.assignment_complete(assignment):
            return assignment

        # Select an unassigned variable
        var = self.select_unassigned_variable(assignment)

        # Try each value in the variable's domain, ordered by the least-constraining values heuristic
        for value in self.order_domain_values(var, assignment):
            # Create a copy of the current assignment to avoid modifying the original
            new_assignment = assignment.copy()

            # Assign the value to the variable
            new_assignment[var] = value

            # If the assignment is consistent, proceed with the next variable
            if self.consistent(new_assignment):
                # Apply arc consistency (optional, for inference)
                inferences = self.ac3([(var, neighbor) for neighbor in self.crossword.neighbors(var)])

                if inferences:  # If arc consistency holds, proceed with backtracking
                    result = self.backtrack(new_assignment)
                    if result is not None:  # If the result is not failure, return it
                        return result

            # If the current value leads to a failure, remove it from the assignment and backtrack

        # If no assignment was found for this variable, return None (failure)
        return None



def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
