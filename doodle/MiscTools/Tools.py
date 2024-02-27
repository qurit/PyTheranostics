from datetime import datetime

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