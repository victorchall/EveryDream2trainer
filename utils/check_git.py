def check_git():
    import subprocess
    try:
        result = subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], capture_output=True, text=True)
        branch = result.stdout.strip()

        result = subprocess.run(["git", "rev-list", "--left-right", "--count", f"origin/{branch}...{branch}"], capture_output=True, text=True)
        ahead, behind = map(int, result.stdout.split())

        if behind > 0:
            print(f"** Your branch '{branch}' is {behind} commit(s) behind the remote.  Consider running 'git pull'.")
        elif ahead > 0:
            print(f"** Your branch '{branch}' is {ahead} commit(s) ahead the remote, consider a pull request.")
        else:
            print(f"** Your branch '{branch}' is up to date with the remote")
    except:
        pass