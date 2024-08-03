import numpy as np

import json
import sys
import traceback

# ANSI escape codes for text color
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def compare_np_torch(np_array, tensor):
    tensor = tensor.detach().numpy()
    assert np_array.shape == tensor.shape, f'numpy: {np_array.shape}, tensor: {tensor.shape}'
    assert np.allclose(np_array, tensor, rtol=1e-05, atol=1e-10), "NOT ALL CLOSE:\n{}\n{}".format(np_array, tensor)
    assert np.abs(np_array - tensor).sum() < 1e-10, "{} vs {}, diff: {}".format(
        np_array, tensor, np.abs(np_array - tensor).sum()
    )
    

def print_failure(cur_test, num_dashes=51):
    print('*' * num_dashes)
    print('The local autograder will not work if you do not pass %s.' % cur_test)
    print('*' * num_dashes)
    print(' ')

def print_name(cur_question):
    print(cur_question)

def print_outcome(short, outcome, point_value, num_dashes=51):
    score = point_value if outcome else 0
    if score != point_value:
        print(RED+"{}: {}/{}".format(short, score, point_value)+RESET)
        print('-' * num_dashes)

def run_tests(tests, summarize=False):
    # calculate number of dashes to print based on max line length
    title = "AUTOGRADER SCORES"
    num_dashes = calculate_num_dashes(tests, title)

    # print title of printout
    print(generate_centered_title(title, num_dashes))

    # Print each test
    scores = {}
    for t in tests:
        if not summarize:
            print_name(t['name'])
        try:
            res = t['handler']()
        except Exception:
            res = False
            traceback.print_exc()
        if not summarize:
            print_outcome(t['autolab'], res, t['value'], num_dashes)
        scores[t['autolab']] = t['value'] if res else 0

    points_available = sum(t['value'] for t in tests)
    points_gotten = sum(scores.values())

    print("\nSummary:")
    pretty_print_scores(scores, points_gotten, points_available)
    print(json.dumps({'scores': scores}))

def calculate_num_dashes(tests, title):
    """Determines how many dashes to print between sections (to be ~pretty~)"""
    # Init based on str lengths in printout
    str_lens = [len(t['name']) for t in tests] + [len(t['autolab']) + 4 for t in tests]
    num_dashes = max(str_lens) + 1

    # Guarantee minimum 5 dashes around title
    if num_dashes < len(title) - 4:
        return len(title) + 10

    # Guarantee even # dashes around title
    if (num_dashes - len(title)) % 2 != 0:
        return num_dashes + 1

    return num_dashes

def generate_centered_title(title, num_dashes):
    """Generates title string, with equal # dashes on both sides"""
    dashes_on_side = int((num_dashes - len(title)) / 2) * "-"
    return dashes_on_side + title + dashes_on_side

def pretty_print_scores(scores, points_gotten, points_available):
    # Determine the maximum length of keys for formatting
    max_key_length = max(len(key) for key in scores)

    # Print the table header
    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')
    print(f'| {"Test".center(max_key_length)} | {"Score".center(11)} |')
    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')

    # Print the table rows
    for key, value in scores.items():
        value_padding = 12 - len(str(value))
        if value == 0: START = RED
        else: START = GREEN
        print('| ' + START + str(key).ljust(max_key_length) + RESET + ' | ' + START + str(value).center(value_padding) + RESET + ' |')

    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')
    print('| ' + 'TOTAL'.center(max_key_length) + ' | ' + f'{points_gotten}/{points_available}'.center(value_padding) + ' |')
    print(f'+{"-" * (max_key_length + 2)}+{"-" * 13}+')