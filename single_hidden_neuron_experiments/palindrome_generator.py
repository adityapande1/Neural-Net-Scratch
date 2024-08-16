import csv


def is_palindrome(string):
    return string == string[::-1]


def generate_data_csv(output_file_path, num_bits=10):
    all_strings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
    ## oversampling
    p_samples = [
        {"string": f"{s}", "target": "P"} for s in all_strings if is_palindrome(s)
    ] * 10
    data = [
        {"string": f"{s}", "target": "P" if is_palindrome(s) else "NP"}
        for s in all_strings
    ]
    data = data + p_samples
    with open(output_file_path, "w", newline="") as csvfile:
        fieldnames = ["string", "target"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    output_file_path = "data.csv"
    generate_data_csv(output_file_path, num_bits=10)
    print(f"Data has been generated and saved to {output_file_path}.")
