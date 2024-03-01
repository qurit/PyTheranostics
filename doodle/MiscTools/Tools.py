from datetime import datetime
import numpy

def hu_to_rho(hu: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a CT array, in HU into a density map in g/cc
    Conversion based on Schneider et al. 2000 (using GATE's material db example)
    """
    # Define the bin edges for HU values
    bins = numpy.array([-1050, -950, -852.884, -755.769, -658.653, -561.538, -464.422, -367.306, -270.191,
                        -173.075, -120, -82, -52, -22, 8, 19, 80, 120, 200, 300, 400, 500, 600, 700, 800,
                        900, 1000, 1100, 1200, 1300, 1400, 1500, 1640, 1807.5, 1975.01, 2142.51, 2300,
                        2467.5, 2635.01, 2802.51, 2970.02, 3000])

    # Define the corresponding density values for each bin
    values = numpy.array([0.00121, 0.102695, 0.202695, 0.302695, 0.402695, 0.502695, 0.602695, 0.702695,
                          0.802695, 0.880021, 0.926911, 0.957382, 0.984277, 1.01117, 1.02955, 1.0616,
                          1.1199, 1.11115, 1.16447, 1.22371, 1.28295, 1.34219, 1.40142, 1.46066, 1.5199,
                          1.57914, 1.63838, 1.69762, 1.75686, 1.8161, 1.87534, 1.94643, 2.03808, 2.13808,
                          2.23808, 2.33509, 2.4321, 2.5321, 2.6321, 2.7321, 2.79105, 2.9])

    # Clip the HU array values to be within the range of defined bins
    hu_clipped = numpy.clip(hu, bins[0], bins[-1])

    # Find the corresponding bin for each HU value
    bin_indices = numpy.digitize(hu_clipped, bins, right=True)

    # Map each bin index to the corresponding density value
    rho = values[bin_indices - 1]

    return rho


def calculate_time_difference(date_str1: str, date_str2: str, 
                              date_format: str = "%Y%m%d %H%M%S") -> float:
    """
    Calculate the time difference in hours between two dates.
    
    The dates should be provided in the "YYYYMMDD HHMMSS" format.
    
    Parameters:
        date_str1 (str): The first date as a string in "YYYYMMDD HHMMSS" format.
        date_str2 (str): The second date as a string in "YYYYMMDD HHMMSS" format.
        date_format (str): The date format of input date_str
        
    Returns:
        float: The difference between the dates in hours.
    """
        
    # Clean up:
    date_str1 = date_str1.split(".")[0]
    date_str2 = date_str2.split(".")[0]

    # Convert string dates to datetime objects
    datetime1 = datetime.strptime(date_str1, date_format)
    datetime2 = datetime.strptime(date_str2, date_format)
    
    # Calculate the difference in hours
    time_diff = datetime1 - datetime2
    hours_diff = time_diff.total_seconds() / 3600
    
    return hours_diff