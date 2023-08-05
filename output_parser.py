import re
import os


def parse_text_file(file_path):
    summed_dict = {
        'best/test/c_acc': 0,
        'best/test/c_f1': 0,
        'best/test/c_loss': 0,
        'best/test/c_prec': 0,
        'best/test/c_rec': 0,
        'best/test/total_loss': 0
    }

    with open(file_path, 'r') as file:
        content = file.read()

        # Use regular expressions to find the relevant keys and values
        pattern = r"'(best/test/c_acc|best/test/c_f1|best/test/c_loss|best/test/c_prec|best/test/c_rec|best/test/total_loss)': (\d+\.?\d*)"
        matches = re.findall(pattern, content)

        for key, value in matches:
            summed_dict[key] += float(value)

    return summed_dict


def output_avg_file(avg_metric_file, output_folder):
    # read files in the folder
    for f in os.listdir(output_folder):
        filename = os.path.join('output_folder', f)
        summed_dict = parse_text_file(filename)

        # append into avg_metric_file
        with open(avg_metric_file, 'a') as file:
            file.write(f'\n******* {f} ********\n\n')
            for key, value in summed_dict.items():
                file.write(f"{key}: {value/5}\n")


if __name__ == '__main__':
    avg_metric_file = 'metrics_average.txt'
    output_folder = 'output_folder'
    output_avg_file(avg_metric_file, output_folder)

    # file_path = 'output_folder/output_1.txt'
    # summed_dict = parse_text_file(file_path)

    # # Print the summed dictionary
    # output_file = 'avg_metrics.txt'
    # with open(output_file, 'a') as file:
    #     for key, value in summed_dict.items():
    #         file.write(f"{key}: {value}\n")
