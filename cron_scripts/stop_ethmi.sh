#!/bin/bash

num=$(ps aux | grep ethminer_frk | grep -v grep | awk '{print $2}'); if [ ! -z $num ]; then kill -9 $num; fi
