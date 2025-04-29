from tabulate import tabulate

def match_and_combine_results(cam_results):
    combined_results = {}
    for cam_result in cam_results:
        for label, quantity in cam_result.items():
            combined_results[label] = max(combined_results.get(label, 0), quantity)
    return combined_results

def count_total_products(combined_results):
    total_bottles = combined_results.get("bottle", 0)
    total_cans = combined_results.get("can", 0)
    return total_bottles, total_cans

def generate_final_output(cam_results):
    combined_results = match_and_combine_results(cam_results)
    total_bottles, total_cans = count_total_products(combined_results)
    return combined_results

def display_results_table(combined_results):
    table = [[label, quantity] for label, quantity in combined_results.items()]
    print(tabulate(table, headers=["Label", "Quantity"], tablefmt="pretty"))