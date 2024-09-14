import math

def compute_decision_margin_distance(act1, act2):
    return signed_distance_to_unit_line(act1, act2)

def signed_distance_to_unit_line(xi, yi):
    '''Calculate the distance between the point (xi, yi) and the line x=y
        distance point (x0,y0) to line (ax + by + c = 0):
        abs(a * x0 + b * y0 + c) / sqrt(a^2 + b^2)
        https://www.mathportal.org/calculators/analytic-geometry/line-point-distance.php
    '''
    distance = (xi-yi) / math.sqrt(2)
    return distance  