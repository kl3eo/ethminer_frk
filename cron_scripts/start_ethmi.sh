#!/bin/bash
sleep 5 && nohup /usr/local/bin/ethminer_frk -G -P http://127.0.0.1:9656 > ~/out.log 2>&1 &
