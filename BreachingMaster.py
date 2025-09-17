import pyautogui
from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import keyboard
import time

# === OCR image to text ===
def extract_text_from_image(pil_image):
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh, config='--psm 6').strip()

def count_buffer_slots(pil_image):
    # Конвертировать PIL -> OpenCV
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Диапазон зелёного цвета в HSV
    lower_green = np.array([30, 80, 80])
    upper_green = np.array([70, 255, 255])


    # Маска по зелёному цвету
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Морфология для соединения пунктирных контуров
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Сохранение для отладки
    debug_path = os.path.join(regions_dir, "debug_closed.png")
    cv2.imwrite(debug_path, closed)

    # Поиск контуров
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Размеры примерно 42x42 с допуском
        if 30 <= w <= 55 and 30 <= h <= 55:
            square_count += 1

    return square_count


# === text to matrix ===
def text_to_matrix(text):
    value_dict = {
    "B": "BD",
    "D": "BD",
    "1": "1C",
    "C": "1C",
    "E": "E9",
    "9": "E9",
    "5": "55",
    "7": "7A",
    "A": "7A",
    "F": "FF",
}
    lines = text.split('\n')
    matrix = []
    for line in lines:
        row = []
        for value in line.split():
            #BD E9 1C 55 7A FF
            final_val = value_dict.get(value[0])
            if final_val is None:
                final_val = value_dict.get(value[1])
            row.append(final_val)
        matrix.append(row)
    return matrix

# === Получение пути к текущему проекту ===
project_dir = os.path.dirname(os.path.abspath(__file__))
regions_dir = os.path.join(project_dir, "regions")
os.makedirs(regions_dir, exist_ok=True)

# === Ожидание нажатия клавиши '9' ===
print("⏳ Ожидание нажатия клавиши '9' для запуска сканирования...")
keyboard.wait('9')
print("📸 Клавиша нажата! Делаю скриншот...")
time.sleep(0.3)

# === Шаг 1: Сделать скриншот и сохранить ===
screenshot_path = os.path.join(project_dir, "full_screen.png")
screenshot = pyautogui.screenshot()
screenshot.save(screenshot_path)

# === Шаг 2: Указать координаты регионов ===
matrix_coords = (410, 556, 980, 1120)
combinations_coords = (1214, 531, 1505, 743)
buffer_coords = (1207, 329, 1671, 392)

# === Шаг 3: Обрезка и сохранение в regions ===
matrix_img = screenshot.crop(matrix_coords)
matrix_img.save(os.path.join(regions_dir, "matrix.png"))

combinations_img = screenshot.crop(combinations_coords)
combinations_img.save(os.path.join(regions_dir, "combinations.png"))

buffer_img = screenshot.crop(buffer_coords)
buffer_img.save(os.path.join(regions_dir, "buffer.png"))


# === Шаг 4: Распознавание ===
matrix_text = extract_text_from_image(matrix_img)
matrix = text_to_matrix(matrix_text)
combinations_text = extract_text_from_image(combinations_img)
combinations = text_to_matrix(combinations_text)
buffer_slots = count_buffer_slots(buffer_img)

# === Шаг 5: Вывод результатов ===
print("\n📦 Матрица байтов:")
print(matrix)

print("\n🧩 Комбинации байтов:")
print(combinations)

print(f"\n📊 Кол-во слотов буфера: {buffer_slots}")
    

# Часть 2 --------------------------------------








# import collections

# # Global variables to store the best result found during the Depth-First Search (DFS)
# # These are used to track the path that completes the most combinations with the fewest buffer slots.
# best_path_coords = []
# max_completed_combs_count = -1
# min_buffer_used_for_max_combs = float('inf')

# def find_subsequences(main_list, sub_list):
#     """
#     Checks if 'sub_list' is a contiguous subsequence within 'main_list'.

#     Args:
#         main_list (list): The list in which to search for the subsequence.
#         sub_list (list): The subsequence to find.

#     Returns:
#         bool: True if sub_list is found as a contiguous subsequence, False otherwise.
#     """
#     n = len(main_list)
#     m = len(sub_list)
#     if m == 0:
#         return True  # An empty subsequence is always present
#     if m > n:
#         return False # Subsequence cannot be longer than the main list
    
#     # Iterate through all possible starting positions for the subsequence in main_list
#     for i in range(n - m + 1):
#         if main_list[i:i+m] == sub_list:
#             return True
#     return False

# def solve_breaching(matrix, combinations, buffer_slots):
#     """
#     Solves the Cyberpunk 2077 Breaching minigame to find the optimal coordinate path.

#     The optimal path is defined as the one that:
#     1. Maximizes the number of completed combinations.
#     2. If multiple paths complete the same maximum number of combinations,
#        it minimizes the number of buffer slots used.

#     Args:
#         matrix (list of list of str): The 2D matrix containing byte values.
#         combinations (list of list of str): A list of target byte combinations to find.
#         buffer_slots (int): The total number of buffer slots available for selections.

#     Returns:
#         list of tuple: A list of (row, col) coordinates representing the optimal path.
#                        Returns an empty list if no valid path can be found.
#     """
#     global best_path_coords, max_completed_combs_count, min_buffer_used_for_max_combs

#     # Reset global state for each call to ensure a fresh start for the solver
#     best_path_coords = []
#     max_completed_combs_count = -1
#     min_buffer_used_for_max_combs = float('inf')

#     rows = len(matrix)
#     cols = len(matrix[0])

#     # Convert combinations to tuples. Tuples are hashable and immutable,
#     # which is efficient for checking membership in sets later.
#     target_combinations = [tuple(c) for c in combinations]

#     def dfs(current_path_coords, current_path_values, current_buffer_used, last_coord, completed_comb_indices_set, next_move_is_column):
#         """
#         Recursive Depth-First Search (DFS) function to explore all possible paths.

#         This function explores paths by making valid moves, tracks completed combinations,
#         and updates the global best path found so far.

#         Args:
#             current_path_coords (list of tuple): A list of (row, col) tuples representing
#                                                  the sequence of selected cells in the current path.
#             current_path_values (list of str): A list of byte values corresponding to
#                                                'current_path_coords'.
#             current_buffer_used (int): The number of buffer slots consumed by 'current_path_coords'.
#             last_coord (tuple): The (row, col) of the most recently selected cell.
#                                  It's None if this is the very first selection.
#             completed_comb_indices_set (set): A set of integer indices, where each index
#                                               corresponds to a combination in 'target_combinations'
#                                               that has been successfully completed by 'current_path_values'.
#             next_move_is_column (bool): True if the next allowed move should be downwards in a column,
#                                         False if it should be rightwards in a row.
#         """
#         global best_path_coords, max_completed_combs_count, min_buffer_used_for_max_combs

#         # 1. Update the overall best result found so far
#         current_completed_count = len(completed_comb_indices_set)
#         if current_completed_count > max_completed_combs_count:
#             # If the current path completes more combinations than any previous best path
#             max_completed_combs_count = current_completed_count
#             min_buffer_used_for_max_combs = current_buffer_used
#             best_path_coords = list(current_path_coords) # Store a copy of the path
#         elif current_completed_count == max_completed_combs_count:
#             # If the current path completes the same number of combinations as the best
#             # then check if it uses fewer buffer slots.
#             if current_buffer_used < min_buffer_used_for_max_combs:
#                 min_buffer_used_for_max_combs = current_buffer_used
#                 best_path_coords = list(current_path_coords) # Store a copy of the path

#         # 2. Base Case (Pruning): Stop exploring this path if all buffer slots are used
#         if current_buffer_used == buffer_slots:
#             return

#         # 3. Generate Next Possible Moves based on alternating rule
#         possible_next_coords = []
        
#         # If it's the very first selection (path is empty)
#         if not current_path_coords:
#             # The first selection MUST be from the first row (row 0)
#             for c in range(cols):
#                 possible_next_coords.append((0, c))
#         else:
#             # For subsequent selections, alternate between column and row moves
#             prev_r, prev_c = last_coord
            
#             if next_move_is_column:
#                 # If next move should be column (down), generate moves downwards in the current column
#                 for r_next in range(prev_r + 1, rows):
#                     possible_next_coords.append((r_next, prev_c))
#             else:
#                 # If next move should be row (right), generate moves rightwards in the current row
#                 for c_next in range(prev_c + 1, cols):
#                     possible_next_coords.append((prev_r, c_next))

#         # 4. Recursive Calls for each valid next move
#         for next_r, next_c in possible_next_coords:
#             # Rule: A cell cannot be selected again within the current path.
#             if (next_r, next_c) in current_path_coords:
#                 continue

#             next_byte = matrix[next_r][next_c]
            
#             # Extend the current path with the new coordinate and its value
#             new_path_coords = current_path_coords + [(next_r, next_c)]
#             new_path_values = current_path_values + [next_byte]
#             new_buffer_used = current_buffer_used + 1
            
#             # Create a new set for completed combinations to avoid modifying the parent's state
#             new_completed_comb_indices_set = set(completed_comb_indices_set) 

#             # Check if adding this new byte completes any previously uncompleted combinations
#             for i, target_comb in enumerate(target_combinations):
#                 if i not in new_completed_comb_indices_set:
#                     if find_subsequences(new_path_values, list(target_comb)):
#                         new_completed_comb_indices_set.add(i)

#             # For the next recursive call, toggle the 'next_move_is_column' flag
#             # (unless it's the very first move, where it's explicitly set to True for the next step)
#             new_next_move_is_column = True if not current_path_coords else not next_move_is_column
            
#             # Recursively call DFS with the updated state
#             dfs(new_path_coords, new_path_values, new_buffer_used, (next_r, next_c), new_completed_comb_indices_set, new_next_move_is_column)

#     # Initiate the DFS. The first call starts with an empty path.
#     # 'next_move_is_column' is initially False, indicating the first selection (from row 0)
#     # is a "row-type" pick, and the *next* step will be a column move.
#     dfs([], [], 0, None, set(), False)

#     # After the DFS completes exploring all possible paths, return the best path found.
#     return best_path_coords

# # Solve the minigame
# solution_path = solve_breaching(matrix, combinations, buffer_slots)

# # Print the result
# print(f"Optimal path found: {solution_path}")
# print(f"Combinations completed: {max_completed_combs_count}")
# print(f"Buffer slots used: {min_buffer_used_for_max_combs}")









def solve_breach(matrix, combinations, buffer_slots):
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    combos = [tuple(combo) for combo in combinations]       # target sequences as tuples for easy comparison
    total_combos = len(combos)
    best_combo_count = 0
    best_paths = []

    def dfs(path, used):
        nonlocal best_combo_count, best_paths
        length = len(path)
        # Construct the byte sequence corresponding to the current path of coordinates
        seq = [matrix[r][c] for (r, c) in path]

        # Check which combos are completed in this sequence
        completed = 0
        completed_flags = [False] * total_combos
        for i, combo in enumerate(combos):
            if not completed_flags[i] and len(combo) <= len(seq):
                # Check if 'combo' appears contiguously in 'seq'
                for start in range(len(seq) - len(combo) + 1):
                    if tuple(seq[start:start+len(combo)]) == combo:
                        completed_flags[i] = True
                        completed += 1
                        break

        # Update best found combos count and paths
        if completed > best_combo_count:
            best_combo_count = completed
            best_paths = [path.copy()]
        elif completed == best_combo_count:
            # Add the path if it's a new optimal path
            best_paths.append(path.copy())

        # If we've used all slots or already completed all combos, stop exploring further
        if length == buffer_slots or completed == total_combos:
            return

        # Prune: estimate the maximum combos possible from this state
        remaining_moves = buffer_slots - length
        # Count how many combos could still be completed (rough optimistic estimate)
        possible_extra = 0
        for i, combo in enumerate(combos):
            if not completed_flags[i]:
                # Minimum moves needed to complete this combo (if started fresh) is its length
                if len(combo) <= remaining_moves:
                    possible_extra += 1
        # If even in the best case the total combos achievable (completed + possible_extra) 
        # is less than current best_combo_count, prune this path
        if completed + possible_extra < best_combo_count:
            return

        # Determine next move positions based on alternating row/col rule
        next_index = length + 1  # the 1-based index of the next move
        if next_index == 1:
            # First move: can pick any position in row 0
            next_positions = [(0, c) for c in range(cols)]
        elif next_index % 2 == 0:
            # Even-indexed move: must pick from the same column as last move
            _, last_col = path[-1]
            next_positions = [(r, last_col) for r in range(rows)]
        else:
            # Odd-indexed move (beyond the first): pick from the same row as last move
            last_row, _ = path[-1]
            next_positions = [(last_row, c) for c in range(cols)]

        # Recurse on all valid next moves
        for (nr, nc) in next_positions:
            if (nr, nc) in used:
                continue  # skip already-used cells to avoid repeats
            used.add((nr, nc))
            path.append((nr, nc))
            dfs(path, used)
            path.pop()
            used.remove((nr, nc))

    # Start DFS from an empty path (will initiate picking from row 0)
    dfs([], set())
    # Output the results. Print all optimal paths and return the first one.
    if best_combo_count == 0:
        print("No valid path found that completes any target combination.")
    else:
        print(f"Maximum combinations completed: {best_combo_count}/{total_combos}")
        print("Optimal paths found:")
        for p in best_paths:
            seq = [matrix[r][c] for (r, c) in p]
            print(f"  Path: {p} -> Sequence: {seq}")
    # Return one of the optimal paths (first one)
    return best_paths[0] if best_paths else []
path = solve_breach(matrix, combinations, buffer_slots)
print("Chosen path:", path)
















print("hello world atak")



temp = [['7A', '7A', 'BD', '7A', '1C', 'E9', 'E9'],
        ['1C', '55', '55', '7A', '1C', '7A', '7A'], 
        ['7A', '55', '55', '1C', '55', '55', '1C'], 
        ['7A', 'FF', '55', '55', '55', '55', '7A'], 
        ['7A', '1C', '7A', 'FF', '7A', '1C', '55'], 
       ['FF', '1C', 'BD', '55', 'E9', 'FF', '7A'], 
       ['1C', 'E9', 'FF', 'BD', 'BD', '1C', '55']]

















# SCREEN CLICKING --------------------------------------------------------------------
matrix_each_coord = {
    "1"
}
def click_at_coordinates(x, y, duration=0.1, clicks=1, interval=0.0):
    """
    Moves the mouse cursor to a specified (x, y) coordinate on the screen
    and performs a click.

    Args:
        x (int): The X-coordinate (horizontal position) on the screen.
        y (int): The Y-coordinate (vertical position) on the screen.
        duration (float): The time in seconds it takes to move the mouse to the coordinates.
                          Set to 0 for instant movement. Default is 0.1 seconds.
        clicks (int): The number of clicks to perform. Default is 1.
        interval (float): The time in seconds between each click (if clicks > 1). Default is 0.0.
    """
    print(f"Moving mouse to ({x}, {y})...")
    # Move the mouse to the specified coordinates
    pyautogui.moveTo(x, y, duration=duration)
    print(f"Clicking {clicks} time(s) at ({x}, {y})...")
    # Perform the click(s)
    pyautogui.click(clicks=clicks, interval=interval)
    print("Click action completed.")
