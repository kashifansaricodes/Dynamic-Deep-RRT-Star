path_not_generated = {0:[]}
image_number = 1
start = 9
goal = 10
if image_number not in path_not_generated:
    path_not_generated[image_number] = [(start, goal)]
else:
    path_not_generated[image_number].append((start, goal))