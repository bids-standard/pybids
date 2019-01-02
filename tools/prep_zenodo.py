#!/usr/bin/env python3
import git
import json
from subprocess import run, PIPE, CalledProcessError
from pathlib import Path
from tempfile import TemporaryDirectory


def decommify(name):
    return ' '.join(name.split(', ')[::-1])


# List of repositories whose commits should be counted as contributions
codependents = ['https://github.com/grabbles/grabbit.git']

# Last shablona commit
origin_commit = 'd72caaf5933907ed699d57faddaec7bfc836ce6f'

git_root = Path(git.Repo('.', search_parent_directories=True).working_dir)
zenodo_file = git_root / '.zenodo.json'

zenodo = json.loads(zenodo_file.read_text()) if zenodo_file.exists() else {}

orig_creators = zenodo.get('creators', [])
creator_map = {decommify(creator['name']): creator
               for creator in orig_creators}

shortlog = run(['git', 'shortlog', '-ns', f'{origin_commit}..'], stdout=PIPE)
counts = [line.split('\t', 1)[::-1]
          for line in shortlog.stdout.decode().split('\n') if line]

# Get additional commit counts from dependencies
with TemporaryDirectory() as tmpdir:
    tmppath = Path(tmpdir)
    for repo in codependents:
        repo_dir = str(tmppath / repo.rsplit('/', 1)[1].split('.', 1)[0])
        try:
            clone = run(['git', 'clone', repo, repo_dir], check=True)
        except CalledProcessError as err:
            raise RuntimeError("Could not clone {}".format(repo)) from err
        tag = run(['git', '-C', repo_dir, 'tag'], stdout=PIPE)
        latest_tag = tag.stdout.decode().strip().rsplit('\n', 1)[1]
        dep_shortlog = run(
            ['git', '-C', repo_dir, 'shortlog', '-ns', latest_tag],
            stdout=PIPE)
        counts.extend(line.split('\t', 1)[::-1]
                      for line in dep_shortlog.stdout.decode().split('\n')
                      if line)

commit_counts = {}
for committer, commits in counts:
    commit_counts[committer] = commit_counts.get(committer, 0) + int(commits)

# Stable sort:
# Number of commits in reverse order
# Ties broken by alphabetical order of first name
committers = [committer
              for committer, _ in sorted(commit_counts.items(),
                                         key=lambda x: (-x[1], x[0]))]

# Tal to the top
first_author = 'Tal Yarkoni'
if committers[0] != first_author:
    committers.remove(first_author)
    committers.insert(0, first_author)

creators = [
    creator_map.get(committer, {'name': committer})
    for committer in committers
    ]

zenodo['creators'] = creators
zenodo_file.write_text(json.dumps(zenodo, indent=2, sort_keys=True) + '\n')
