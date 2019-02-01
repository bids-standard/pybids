#!/bin/bash
#
# Collects the pull-requests since the latest release and
# aranges them in the CHANGES.txt file.
#
# This is a script to be run before releasing a new version.
#
# Usage /bin/bash update_changes.sh 1.0.1
#
# Adapted from https://github.com/nipy/nipype/blob/98beb0a/tools/update_changes.sh

# Setting      # $ help set
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

CHANGES=CHANGELOG.md

# Elaborate today's release header
HEADER="## Version $1 ($(date '+%B %d, %Y'))"
cat <<END > newchanges
# Changelog

## Version $1 ($(date '+%B %d, %Y'))

END

# Search for PRs since previous release
git log --grep="Merge pull request" `git describe --tags --abbrev=0`..HEAD --pretty='format:* %b %s' | sed  's+Merge pull request \(\#[^\d]*\)\ from\ .*+(\1)+' >> newchanges

# Append old changes
tail -n+2 $CHANGES >> newchanges

# Replace old CHANGES with new file
mv newchanges $CHANGES

