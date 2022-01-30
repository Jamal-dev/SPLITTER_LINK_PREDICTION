import subprocess

with open('train_splitter_commands.txt', 'r') as f:
    lines = f.readlines()
    commands = [line.strip() for line in lines]

command = ' '.join(commands)

ret = subprocess.run(command, capture_output=True, shell=True)
print(ret.stdout.decode())
