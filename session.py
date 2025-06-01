# session.py
#!/usr/bin/env python3
import subprocess
import argparse

# python session.py --session my_sim_session


def run_in_tmux(session_name='sim_session', command='python main.py'):
    """
    Create or attach to a tmux session and run the specified command.
    """
    # Check if the session already exists
    try:
        subprocess.run(['tmux', 'has-session', '-t', session_name],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Session '{session_name}' exists. Attaching...")
    except subprocess.CalledProcessError:
        # Create a new detached session
        subprocess.run(['tmux', 'new-session', '-d', '-s', session_name, command],
                       check=True)
        print(f"Created tmux session '{session_name}' and started: {command}")
    # Finally attach
    subprocess.run(['tmux', 'attach-session', '-t', session_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run simulation inside a tmux session'
    )
    parser.add_argument('--session', '-s', default='sim_session',
                        help='Name of the tmux session')
    parser.add_argument('cmd', nargs=argparse.REMAINDER,
                        help='Command to run inside tmux (default: python main.py)')
    args = parser.parse_args()
    cmd = ' '.join(args.cmd) if args.cmd else 'python main.py'
    run_in_tmux(session_name=args.session, command=cmd)