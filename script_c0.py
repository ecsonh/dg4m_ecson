import subprocess
import os


def execute_command(command, output_file):
    # Execute the command and redirect stdout to the output file
    os.system(f"{command} | tee -a {output_file}")


def experiment_combination(domain, seed):
    output_folder = "output_folder"
    os.makedirs(output_folder, exist_ok=True)
    for d in domain:
        # write every 5 seeds into one file
        for s in seed:
            command = "python3 discovhar_DANN.py --dataset dsads --model dsads64 --batch_size 256 --learning_rate 0.08 --temp 0.07 --epochs 2 --classifier_epochs 2 --cuda_device 0 --LODO %s --weighted_supcon 1 --seed %s" % (
                d, s)
            output_file = os.path.join(
                output_folder, f"output_lodo_{d}.txt")
            execute_command(command, output_file)

    return

# test/c_acc, test/c_f1, test/c_loss, test/c_prec, test/c_rec, test/total_loss


if __name__ == '__main__':
    domain = [0, 1]
    seed = [0, 1, 2, 3, 4]
    loro_loao = [(0, 1), (10, 0.9), (15, 0.75), (20, 0.65), (30, 0.55)]
    experiment_combination(domain, seed)
