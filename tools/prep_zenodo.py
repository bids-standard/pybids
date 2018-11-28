#!/usr/bin/env python3
import os
import git
import json
from subprocess import run, PIPE
from pathlib import Path
from pprint import pprint

# Last shablona commit
origin_commit = 'd72caaf5933907ed699d57faddaec7bfc836ce6f'

git_root = Path(git.Repo('.', search_parent_directories=True).working_dir)
zenodo_file = git_root / '.zenodo.json'

zenodo = json.loads(zenodo_file.read_text()) if zenodo_file.exists() else {}

orig_creators = zenodo.get('creators', [])
creator_map = {creator['name']: creator for creator in orig_creators}

shortlog = run(['git', 'shortlog', '-ns', f'{origin_commit}..'], stdout=PIPE)
committers = [line.split('\t', 1)[1]
              for line in shortlog.stdout.decode().split('\n') if line]

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
