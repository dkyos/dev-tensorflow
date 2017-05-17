#!/usr/bin/env python

"""Identifies the landmark for the given image."""

import argparse

from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials


DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'


def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1', credentials=credentials,
            discoveryServiceUrl=DISCOVERY_URL)


def identify_landmark(gcs_uri, max_results=4):
    """Uses the Vision API to identify the landmark in the given image.
    Args:
        gcs_uri: A uri of the form: gs://bucket/object
    Returns:
        An array of dicts with information about the landmarks in the picture.
    """
    batch_request = [{
        'image': {
            'source': {
                'gcs_image_uri': gcs_uri
            }
        },
        'features': [{
            'type': 'LANDMARK_DETECTION',
            'maxResults': max_results,
        }]
    }]

    service = get_vision_service()
    request = service.images().annotate(body={
            'requests': batch_request,
            })
    response = request.execute()

    return response['responses'][0].get('landmarkAnnotations', None)


def main(gcs_uri):
    if gcs_uri[:5] != 'gs://':
        raise Exception('Image uri must be of the form gs://bucket/object')
        annotations = identify_landmark(gcs_uri)
        if not annotations:
            print('No landmark identified')
        else:
            print('\n'.join(a['description'] for a in annotations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Identifies the landmark in the given image.')
    parser.add_argument(
        'gcs_uri', help=('The Google Cloud Storage uri to the image to identify'
            ', of the form: gs://bucket_name/object_name.jpg'))
    args = parser.parse_args()
    main(args.gcs_uri)
