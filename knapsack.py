import numpy as np

'''
------------------------------------------------
Use dynamic programming (DP) to solve 0/1 knapsack problem
Time complexity: O(nW), where n is number of items and W is capacity

knapsack_dp(values, weights, n_items, capacity, return_all=False)

Input arguments:
  1. values: a list of numbers in either int or float
  2. weights: a list of int numbers specifying weights of items
  3. n_items: an int number indicating number of items
  4. capacity: an int number indicating the knapsack capacity
  5. return_all: whether return all info, default is False

Return:
  1. picks: a list of positions of selected items
  2. max_val: maximum value (optional)
------------------------------------------------
'''


def knapsack_dp(values, weights, n_items, capacity, return_all=False):

    check_inputs(values, weights, n_items, capacity)
    table = np.zeros((n_items + 1, capacity + 1),dtype=np.float32)
    keep = np.zeros((n_items + 1, capacity + 1),dtype=np.float32)

    # Dynamic Programming
    for i in range(1, n_items + 1):
        for w in range(0, capacity + 1):
            wi = weights[i - 1]
            vi = values[i - 1]

            if (wi <= w) and (vi + table[i - 1, w - wi] > table[i - 1, w]):
                table[i, w] = vi + table[i - 1, w - wi]
                keep[i, w] = 1
            else:
                table[i, w] = table[i - 1, w]

    # Find selected items
    picks = []
    K = capacity

    for i in range(n_items, 0, -1):
        if keep[i, K] == 1:
            picks.append(i)
            K -= weights[i - 1]

    # Sort selected items
    picks.sort()
    # Convert to 0-based indexing
    picks = [x - 1 for x in picks]

    if return_all:
        max_val = table[n_items, capacity]
        return picks, max_val

    return picks


def check_inputs(values, weights, n_items, capacity):
    # Check variable types
    assert isinstance(values, list)
    assert isinstance(weights, list)
    assert isinstance(n_items, int)
    assert isinstance(capacity, int)
    # Check value types
    assert all(isinstance(val, int) or isinstance(val, float) for val in values)
    assert all(isinstance(val, int) for val in weights)
    # Check validity of weights
    assert all( val >= 0 for val in weights)
    assert n_items > 0
    assert capacity > 0


# Main program
if __name__ == '__main__':
    values = [2, 3, 4]
    weights = [1, 2, 3]
    n_items = 3
    capacity = 3
    picks = knapsack_dp(values, weights,n_items,capacity)
    print("Selected item indices:", picks)