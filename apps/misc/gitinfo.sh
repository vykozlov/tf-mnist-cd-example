#!/bin/bash
###########################
#
# Basic script to get info
# about last git commit
#
##########################
GitDir=$(dirname $0)
GitCommit=$(cd $GitDir && git rev-parse HEAD)
GitAuthor=$(cd $GitDir && git log -1 |grep "Author:")
GitAuthor=${GitAuthor//Author:/GitAuthor,}
GitDate=$(cd $GitDir && git log -1 |grep "Date:")
GitDate=${GitDate//Date:/GitDate,}
echo "GitCommit, $GitCommit"
echo $GitAuthor
echo $GitDate

