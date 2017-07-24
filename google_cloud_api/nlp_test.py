#!/usr/bin/env python

# Copyright 2017 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import base64
import json
import io
import os
import readline
import time
import ffmpy
import httplib2
import pprint
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from google.cloud import translate
from pick import pick
from termcolor import colored
import sounddevice as sd
import scipy.io.wavfile as scipy
from pygments import highlight, lexers, formatters
import argparse
from google.cloud import language


# Audio recording duration and sample rate
DURATION = 5
SAMPLE_RATE = 16000
# Languages supported by Neural Machine Translation
SUPPORTED_LANGUAGES = {"German": "de", "Spanish": "es", "French": "fr",
                       "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
                       "Turkish": "tr", "Chinese(Simplified)": "zh-CN"}
# [START authenticating]
DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
                 'version={apiVersion}')
# Application default credentials provided by env variable
# GOOGLE_APPLICATION_CREDENTIALS


def get_service(api, version):
    credentials = GoogleCredentials.get_application_default().create_scoped(
        ['https://www.googleapis.com/auth/cloud-platform'])
    http = httplib2.Http()
    credentials.authorize(http)
    return discovery.build(
        api, version, http=http, discoveryServiceUrl=DISCOVERY_URL)
# [END authenticating]

def call_nl_api(text):
    # We use v1beta2 version api  for korean support 
    service = get_service('language', 'v1beta2')
    service_request = service.documents().annotateText(
        body={
            'document': {
                'type': 'PLAIN_TEXT',
                'language' : 'ko',
                'content': text,
            },
            'features': {
                "extractSyntax": True,
                "extractEntities": True,
                "extractDocumentSentiment": True,
            }
        }
    )
    response = service_request.execute()

    print(colored("\nHere's the JSON repsonse" +
                  "for one token of your text:\n",
                  "cyan"))
    formatted_json = json.dumps(response['tokens'][0], indent=2)

    colorful_json = highlight(formatted_json,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print(colorful_json)

    score = response['documentSentiment']['score']
    output_text = colored(analyze_sentiment(score), "cyan")

    pprint.pprint(response['sentences'])
    # magnitude: A non-negative number in the [0, +inf) range, which represents the absolute magnitude of sentiment regardless of score (positive or negative).
    # Score: Sentiment score between -1.0 (negative sentiment) and 1.0 (positive sentiment).

    if response['entities']:
        entities = str(analyze_entities(response['entities']))
        output_text += colored("\nEntities found: " + entities, "cyan")
    return [output_text, response['language']]


def analyze_sentiment(score):
    sentiment_str = "You seem "
    if -1 <= score < -0.5:
        sentiment_str += "angry. Hope you feel better soon!"
    elif -0.5 <= score < 0.5:
        sentiment_str += "pretty neutral."
    else:
        sentiment_str += "very happy! Yay :)"
    return sentiment_str + "\n"


def analyze_entities(entities):
    arr = []
    for entity in entities:
        if 'wikipedia_url' in entity['metadata']:
            arr.append(entity['name'] + ': ' +
                       entity['metadata']['wikipedia_url'])
        else:
            arr.append(entity['name'])
    return arr


def handle_nl_call(text):
    nl_response = call_nl_api(text)
    analyzed_text = nl_response[0]
    print(analyzed_text)

NL_TEXT = "스마트배송이라 빠르고 편리하네요. 리엔샴푸뽁뽁이로 넘 깔끔하게 배송되었습니다. 지성두피인데 사용해보니 머리카락이 약간 묵직해지는 느낌입니다. 탈모예방에 도움이 될것같아요."

#NL_TEXT = "Because it is smart shipping, it is fast and convenient. Rien shampoo has been shipped in a neat and clean. I have an oily scalp but when I use it, it feels a bit heavier. I think it will help prevent hair loss."
handle_nl_call(NL_TEXT)
