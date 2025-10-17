# Harvard CS50's Introduction to Artificial Intelligence with Python

This repository contains all of my project implementations for Harvard's CS50's Introduction to Artificial Intelligence with Python course. Each project explores fundamental concepts in artificial intelligence, including search algorithms, knowledge representation, uncertainty, optimization, machine learning, neural networks, and natural language processing.
CS50's Introduction to Artificial Intelligence with Python explores the concepts and algorithms at the foundation of modern artificial intelligence, diving into the ideas that give rise to technologies like game-playing engines, handwriting recognition, and machine translation. 
Through hands-on projects, I gained exposure to the theory behind graph search algorithms, classification, optimization, reinforcement learning, and other topics in artificial intelligence as I incorporated them into my own Python programs.

## Projects

### Week 0: Search

#### Degrees
A program that determines the "degrees of separation" between two actors based on the movies they've starred in, similar to the "Six Degrees of Kevin Bacon" game. The program uses breadth-first search to find the shortest path connecting any two actors through the movies they've appeared in together.  
**Key Concepts:** Graph search, BFS, state space representation

#### Tic-Tac-Toe
An AI that plays Tic-Tac-Toe optimally using the Minimax algorithm. The AI considers all possible game states to determine the optimal move, ensuring it never loses when playing optimally. The implementation includes alpha-beta pruning for improved efficiency.  
**Key Concepts:** Game theory, minimax, adversarial search, optimal play

### Week 1: Knowledge

#### Knights
A program that solves "Knights and Knaves" logic puzzles using propositional logic. Knights always tell the truth, while Knaves always lie. The AI uses model checking to determine which characters are knights and which are knaves based on their statements.  
**Key Concepts:** Propositional logic, knowledge representation, model checking, logical inference

#### Minesweeper
An AI that plays Minesweeper by making logical inferences about mine locations. The AI uses knowledge representation to track what it knows about the board and makes safe moves by reasoning about cell constraints. It implements arc consistency and inference to determine which cells are safe and which contain mines.  
**Key Concepts:** Knowledge-based agents, logical inference, constraint satisfaction

### Week 2: Uncertainty

#### PageRank
An implementation of Google's PageRank algorithm that ranks web pages by importance. The program uses both a random surfer model (sampling from a Markov chain) and an iterative algorithm to calculate PageRank values, demonstrating how probability theory can be applied to information retrieval.  
**Key Concepts:** Markov chains, probability distributions, random walks, iterative algorithms

#### Heredity
An AI that assesses the likelihood of a person having a particular genetic trait based on family relationships and observed traits. The program uses Bayesian networks to model gene inheritance and calculates probability distributions using joint probability and inference by enumeration.  
**Key Concepts:** Bayesian networks, joint probability, conditional probability, genetic inheritance modeling

### Week 3: Optimization

#### Crossword
An AI that generates crossword puzzles by treating the problem as a constraint satisfaction problem. The program uses backtracking search with arc consistency (AC-3 algorithm) to efficiently find valid crossword solutions that satisfy all unary and binary constraints.  
**Key Concepts:** CSPs, backtracking search, arc consistency, heuristics (MRV, degree heuristic, least-constraining values)

### Week 4: Learning

#### Shopping
A machine learning model that predicts whether online shopping customers will complete a purchase based on their browsing behavior. The program uses a k-nearest neighbors classifier trained on about 12,000 user sessions, measuring performance through sensitivity and specificity metrics.  
**Key Concepts:** Supervised learning, k-NN classification, feature engineering, model evaluation

#### Nim
An AI that teaches itself to play the game of Nim through reinforcement learning. Using Q-learning, the AI learns optimal strategies by playing thousands of games against itself, updating Q-values based on rewards and future reward estimates.  
**Key Concepts:** Reinforcement learning, Q-learning, exploration vs exploitation, epsilon-greedy algorithm

### Week 5: Neural Networks

#### Traffic
A neural network that classifies road signs from images using TensorFlow and the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The program experiments with different CNN architectures, including varying numbers of convolutional layers, pooling layers, hidden layers, and dropout to achieve high accuracy in traffic sign classification.  
**Key Concepts:** CNNs, computer vision, image classification, deep learning, TensorFlow/Keras

### Week 6: Language

#### Parser
A program that parses English sentences using context-free grammar rules and extracts noun phrase chunks. The parser uses the NLTK library to tokenize sentences and applies grammar rules to determine sentence structure, helping computers understand the syntactic relationships between words.  
**Key Concepts:** NLP, context-free grammars, parsing, syntax trees, tokenization

#### Attention
An exploration of BERT's attention mechanism for masked language modeling. The program uses a pre-trained BERT model to predict masked words in sentences and generates attention diagrams for all 144 attention heads (12 layers × 12 heads). The project includes analysis of what linguistic patterns different attention heads learn to recognise.  
**Key Concepts:** Transformers, attention mechanisms, BERT, masked language modeling, language understanding

## Technologies Used

- **Python 3.12**
- **Libraries:** 
  - NLTK (Natural Language Toolkit)
  - TensorFlow/Keras
  - scikit-learn
  - OpenCV (cv2)
  - Transformers (Hugging Face)
  - Pygame
  - Pillow

## Running the Projects
Each project directory contains its own implementation files and can be run independently

