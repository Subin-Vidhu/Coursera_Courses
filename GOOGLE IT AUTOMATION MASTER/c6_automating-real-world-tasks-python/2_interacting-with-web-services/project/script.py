#! /usr/bin/env python3


import os
import requests


dir = "/data/feedback/"
for file in os.listdir("/data/feedback/"):

    format = ["title", "name", "date", "feedback"]
    content = {}

    with open('{}/{}'.format(dir, file), 'r') as txtfile:
        counter = 0
        for line in txtfile:
            line = line.replace("\n", "")
            content[format[counter]] = line.strip('\n')
            counter += 1

    response = requests.post("http://34.29.168.103/feedback/", json=content)
    if not response.ok:
        raise Exception("POST failed! | Status code: {} | File: {}".format(
            response.status_code, file))
    print("Feedback added!")
