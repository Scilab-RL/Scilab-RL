import os


def test_error_logs_in_dir(logs_path):
    print("Performing error testing.")
    errors = []
    for filename in sorted(os.listdir(logs_path)):
        if filename[-8:] == "_err.log":
            f_path = os.path.join(logs_path, filename)
            with open(f_path, 'r') as f:
                file_content = f.read().lower()
                if "error" in file_content:
                    print("Error. See file {}".format(filename))
                    errors.append(filename)

    print("Error testing results: {} errors found.".format(len(errors)))


if __name__ == "__main__":
    logs_path = "test_logs"
    test_error_logs_in_dir(logs_path)