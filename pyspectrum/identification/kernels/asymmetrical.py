import numpy as np

def asymmetrical_rect_zero_area(x, mean, width):
    """
    non-symmetrical zero area function for non-symmetrical peak recognition

    Parameters
    ----------
    x: array-like
        Input values.
    mean: float
        Mean (center) of the Gaussian.
    width: float
        width of the rect
    Returns
    -------
    numpy array

1   1-----1         1
1   1     1---------1
1   1
1----
    """
    left_right_width_factor = 0.7
    func_val = []
    domains = [mean - width/2-round(left_right_width_factor*2*width),
               mean - width / 2,
               mean + width / 2,
               mean + width / 2 + round((1-left_right_width_factor)*2*width)
               ]
    height_center = 2
    height_left = -(domains[2]-domains[1])/(domains[1]-domains[0])
    height_right = -(domains[2]-domains[1])/(domains[3]-domains[2])
    for domain_val in x:
        # if it's from the left
        if domain_val <= domains[0]:
            func_val.append(0)
        # if it's from the right
        elif domain_val > domains[3]:
            func_val.append(0)
        elif domains[0] < domain_val <= domains[1]:
            func_val.append(height_left)
        elif domains[1] < domain_val <= domains[2]:
            func_val.append(height_center)
        # last option is center_width_domain[1] <= domain_val <= right_width_domain
        else:
            func_val.append(height_right)
    return np.array(func_val)
