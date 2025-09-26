import os
import time

import tqdm
import requests


text_path = './system_outputs/signsuisse_test/test.txt'

language_map = {
    'dsgs': {
        'source': 'de',
        'target': 'sgg',
    },
    'lsf-ch': {
        'source': 'fr',
        'target': 'ssr',
    },
    'lis-ch': {
        'source': 'it',
        'target': 'slf',
    },
}

with open(text_path) as file:
    for index, line in tqdm.tqdm(enumerate(file)):
        pose_path = f'./system_outputs/signsuisse_test/sign_mt_v2/{index}.raw.pose'

        if os.path.exists(pose_path):
            continue

        sent = line.rstrip()
        language = sent.split(' ')[0][1:-1]
        sent = ' '.join(sent.split(' ')[1:])

        print(language)
        print(sent)

        # Define the URL, parameters, and headers
        url = "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
        params = {
            "text": sent,
            "spoken": language_map[language]['source'],
            "signed": language_map[language]['target'],
            "full_keypoints": 1,
            "no_fingerspelling": 1,
        }
        headers = {
            "Authorization": "Bearer Debug_3Zk1iR9ew1EmabCogstDXkG8",
            "Content-Type": "application/pose"
        }
        print(params)

        # Make the GET request
        response = requests.get(url, params=params, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the result to a file
            with open(pose_path, "wb") as file:
                file.write(response.content)
            print(f"Response saved to {pose_path}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

        time.sleep(1)