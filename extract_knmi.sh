#!/bin/bash

# Use this script with a raw data list URL from KNMI and it will download all
# the linked files.

curl -s $1 | grep -o '"data/.\+dat"' | sed 's/\"//g' | sed 's/^/http:\/\/climexp.knmi.nl\//' | xargs wget
