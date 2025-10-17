import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialize the probability distribution dictionary
    probability_distribution = {}
    
    # Get the set of links from the current page
    links = corpus.get(page, set())
    
    # Number of pages in the corpus
    num_pages = len(corpus)
    
    if links:
        # Number of links on the current page
        num_links = len(links)
        
        # Probability of choosing a page from the links on the current page
        for p in corpus:
            probability_distribution[p] = (1 - damping_factor) / num_pages
        
        for link in links:
            probability_distribution[link] += damping_factor / num_links
    else:
        # If no links, assume the page links to all pages in the corpus, including itself
        for p in corpus:
            probability_distribution[p] = 1 / num_pages
    
    return probability_distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize the dictionary to store the counts of each page
    page_counts = {page: 0 for page in corpus}

    # Generate random start page from corpus
    current_page = random.choice(list(corpus.keys()))

    for i in range (n):
        # Count the current page
        page_counts[current_page] += 1

        # Get the transition model for the current page
        probabilities = transition_model(corpus, current_page, damping_factor)

        # Choose next page based on transition probabilities 
        current_page = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))

    # Convert counts to probabilities 
    s_pagerank = {page: count / n for page, count in page_counts.items()}

    return s_pagerank




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)

    # Initialize the dictionary to store the pagerank of each page
    iter_pagerank = {page: 1/n for page in corpus}

    # Initialize a variable to check for convergence
    converged = False

    while not converged:
        new_pagerank = {}

        # Iterate over each page to update its PageRank
        for page in corpus:
            # Start with the damping factor component
            rank = (1 - damping_factor) / n

            # Sum over all pages that link to the current page
            for i in corpus:
                if page in corpus[i]:
                    rank += damping_factor * iter_pagerank[i] / len(corpus[i])
                elif len(corpus[i]) == 0:
                    # Handle the case where i has no outgoing links
                    rank += damping_factor * iter_pagerank[i] / n

            new_pagerank[page] = rank

        # Check for convergence: if all changes are less than 0.001
        converged = all(abs(new_pagerank[page] - iter_pagerank[page]) < 0.001 for page in corpus)

        # Update PageRank values
        iter_pagerank = new_pagerank

    # Normalize the PageRank values to ensure they sum to 1
    total_sum = sum(iter_pagerank.values())
    iter_pagerank = {page: rank / total_sum for page, rank in iter_pagerank.items()}

    return iter_pagerank


if __name__ == "__main__":
    main()
