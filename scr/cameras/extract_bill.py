from tabulate import tabulate

def match_and_combine_results(cam_results):
    """
    Match and combine results from multiple cameras without duplicating identical detections.
    If a label appears in multiple cameras, take the highest quantity.
    
    Args:
        cam_results (list): List of dictionaries, where each dictionary contains the detection results from a camera.
                            Each dictionary has the format {"label": quantity}.
    
    Returns:
        dict: A combined dictionary with labels and their highest detected quantities.
    """
    combined_results = {}
    
    for cam_result in cam_results:
        for label, quantity in cam_result.items():
            if label in combined_results:
                combined_results[label] = max(combined_results[label], quantity)
            else:
                combined_results[label] = quantity
    
    return combined_results

def count_total_products(combined_results):
    """
    Count the total number of products (bottles and cans) from the combined results.
    
    Args:
        combined_results (dict): A dictionary with labels and their quantities.
    
    Returns:
        tuple: (total_bottles, total_cans)
    """
    total_bottles = combined_results.get("bottle", 0)
    total_cans = combined_results.get("can", 0)
    return total_bottles, total_cans

def check_quantities(combined_results, total_bottles, total_cans):
    """
    Check if the sum of all detected products matches the sum of bottles and cans.
    
    Args:
        combined_results (dict): A dictionary with labels and their quantities.
        total_bottles (int): Total number of bottles.
        total_cans (int): Total number of cans.
    
    Returns:
        bool: True if the quantities match, False otherwise.
    """
    total_products = total_bottles + total_cans
    total_combined = sum(combined_results.values())
    
    if total_products != total_combined:
        print(f"Warning: Quantities do not match! Total products: {total_products}, Combined quantities: {total_combined}")
        return False
    return True

def generate_final_output(cam_results):
    """
    Generate the final output by matching results, counting products, and checking quantities.
    
    Args:
        cam_results (list): List of dictionaries, where each dictionary contains the detection results from a camera.
    
    Returns:
        dict: The final combined dictionary with matched labels and their highest quantities.
    """
    # Step 1: Match and combine results
    combined_results = match_and_combine_results(cam_results)
    
    # Step 2: Count total bottles and cans
    total_bottles, total_cans = count_total_products(combined_results)
    
    # Step 3: Check if quantities match
    check_quantities(combined_results, total_bottles, total_cans)
    
    return combined_results

def display_results_table(combined_results):
    """
    Display the results in a table format.
    
    Args:
        combined_results (dict): A dictionary with labels and their quantities.
    """
    table = []
    for label, quantity in combined_results.items():
        table.append([label, quantity])
    
    print(tabulate(table, headers=["Label", "Quantity"], tablefmt="pretty"))
