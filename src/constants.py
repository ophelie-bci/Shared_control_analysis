# Mapping of 3D target coordinates to direction labels
target_mapping = {
    (-7, 0.75, 6): 'left',
    (-3.5, 0.75, 8.5): 'slight_left',
    (0, 0.75, 9.2): 'straight',
    (3.5, 0.75, 8.5): 'slight_right',
    (7, 0.75, 6): 'right'
}
# target_mapping = {
#     (5,0,14.5): 'left', 
#     (-5,0,14.5): 'right'}

# Supported monkey names
monkey_names = ["Monkey 3", "Monkey 1"]

# Supported experiment names
experiment_names = [
    "AI Obstacle",
    "AI Appearing Obstacle",
    "AI Respawn"
]
target_to_obstacle_mapping = {
    (-7.0, 6.0): (-3.5, 3),
    (-3.5, 8.5): (-1.75, 4.25),
    (0.0, 9.2): (0.0, 4.6),
    (3.5, 8.5): (1.75, 4.25),
    (7.0, 6.0): (3.5, 3)
}

CENTRAL_TARGETS = {'straight', 'slight_left', 'slight_right'}
PERIPHERAL_TARGETS = {'left', 'right'}
