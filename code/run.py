import sys
import subprocess
import argparse
from os.path import join as fullfile

parser = argparse.ArgumentParser(description='')

# Exp config
parser.add_argument('--exp_path', default='../exps', type=str)
parser.add_argument('--exp', default=['main'], type=str, nargs='+')

# Training related config
parser.add_argument('--training', action='store_true')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--recover', action='store_true')
parser.add_argument('--workers', default=4, type=int)

# HPC related config
parser.add_argument('--HPC', action='store_true')
parser.add_argument('--HPC2', action='store_true')
parser.add_argument('--num_jobs', default=1, type=int)
parser.add_argument('--job_nos', default=[0], type=int, nargs='+')
parser.add_argument('--config_file', default='run.sh', type=str)
parser.add_argument('--python', default='python', type=str)
parser.add_argument('--parallel', action='store_true')

args = parser.parse_args()


def generate_cmd(exp):
    common_flags = ' --exp_path {exp_path} --exp {exp}{training}{finetune}{recover} --workers {workers}'.format(
        exp_path=args.exp_path,
        exp = exp,
        training=' --training' if args.training else '',
        finetune=' --finetune' if args.finetune else '',
        recover=' --recover' if args.recover else '',
        workers=args.workers
    )

    python = args.python
    cmd = python + ' main.py' + common_flags

    return cmd


def group_cmds(cmds, n):
    part_size = len(cmds) // n
    remainder = len(cmds) % n

    groups = []
    start = 0

    for i in range(n):
        end = start + part_size + (1 if i < remainder else 0)
        groups.append(cmds[start:end])
        start = end

    return groups


def main():
    if args.HPC or args.HPC2:
        # Load HPC config file and find the line related to job name
        import re
        base_config_file = args.config_file
        with open(args.config_file) as f:
            base_config = f.readlines()
            line_job_name = 0
            for i, line in enumerate(base_config):
                if args.HPC:
                    match = re.search(r'#BSUB -J .+', line)
                else:
                    match = re.search(r'#SBATCH -J .+', line)
                if match:
                    line_job_name = i
                    break
    if len(args.job_nos) == 0:
        args.job_nos = list(range(args.num_jobs))

    cmds = []
    for exp in args.exp:
        print(f'Add task: {exp}')
        cmd = generate_cmd(exp)
        cmds.append(cmd)

    cmd_groups = group_cmds(cmds, args.num_jobs)
    for job_no in args.job_nos:
        if job_no >= len(cmd_groups):
            continue
        cmd_group = cmd_groups[job_no]
        if args.HPC or args.HPC2:
            # Set the job name based on exp name
            if args.HPC:
                base_config[line_job_name] = f'#BSUB -J {exp.replace("/", "+")}' + ('' if args.num_jobs == 1 else f'({job_no + 1}/{args.num_jobs})')
            else:
                base_config[line_job_name] = f'#SBATCH -J {exp.replace("/", "+")}' + ('' if args.num_jobs == 1 else f'({job_no + 1}/{args.num_jobs})')
            config_file = base_config_file + '.tmp'
            with open(config_file, 'w') as f:
                f.write(''.join(base_config))
                if len(cmd_group) == 1:
                    f.write(cmd_group[0])
                else:
                    for cmd in cmd_group:
                        f.write(f'\n{cmd}')
                        if args.parallel:
                            f.write(' &')
                    if args.parallel:
                        f.write('\nwait')
            cmd_submit = 'bsub < ' if args.HPC else 'sbatch '
            p = subprocess.Popen(cmd_submit + config_file, shell=True, stdout=sys.stdout, stderr=sys.stderr)
            p.wait()
        else:
            processes = []
            for cmd in cmd_group:
                p = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr)
                if args.parallel:
                    processes.append(p)
                else:
                    p.wait()
            if args.parallel:
                for p in processes:
                    p.wait()


if __name__ == '__main__':
    main()
