#!/bin/bash
#
# Collects the pull-requests since the latest release and
# arranges them in the CHANGELOG.rst file.
#
# This is a script to be run before releasing a new version.
#
# Usage /bin/bash update_changes.sh 1.0.1
#
# Adapted from https://github.com/nipy/nipype/blob/98beb0a/tools/update_changes.sh

# Setting      # $ help set
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

ROOT=$( git rev-parse --show-toplevel )
CHANGES=$ROOT/CHANGELOG.rst

# Elaborate today's release header
HEADER="Version $1 ($(date '+%B %d, %Y'))"
UNDERLINE=$( printf "%${#HEADER}s" | tr " " "-" )
cat <<END > newchanges
Changelog
=========

$HEADER
$UNDERLINE

END

# Search for PRs since previous release
MERGE_COMMITS=$( git log --grep="Merge pull request\|(#.*)$" `git describe --tags --abbrev=0`..HEAD --pretty='format:%h' )
for COMMIT in ${MERGE_COMMITS//\n}; do
    SUB=$( git log -n 1 --pretty="format:%s" $COMMIT )
    if ( echo $SUB | grep "^Merge pull request" ); then
        # Merge commit
        PR=$( echo $SUB | sed -e "s/Merge pull request \#\([0-9]*\).*/\1/" )
        TITLE=$( git log -n 1 --pretty="format:%b" $COMMIT )
    else
        # Squashed merge
        PR=$( echo $SUB | sed -e "s/.*(\#\([0-9]*\))$/\1/" )
        TITLE=$( echo $SUB | sed -e "s/\(.*\)(\#[0-9]*)$/\1/" )
    fi
    echo "* $TITLE (#$PR)" >> newchanges
done

# Append old changes
tail -n+3 $CHANGES >> newchanges

# Replace old CHANGES with new file
mv newchanges $CHANGES
