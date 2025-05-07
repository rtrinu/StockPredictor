def simple_averages(rf,hybrid):
    average = (rf + hybrid)/2
    return average

def weighted_averages(rf,hybrid):
    weighted_average = (rf * 0.25) + (hybrid * 0.75 )
    return weighted_average