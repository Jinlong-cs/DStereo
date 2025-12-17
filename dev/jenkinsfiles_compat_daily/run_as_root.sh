#!/usr/bin/expect

set func [lindex $argv 0]
spawn su root
expect "Password:"
send "\r"
expect "#"
send "sh ${func} \r"
interact
