from dask.distributed import Client, get_client, Variable, fire_and_forget
import numpy as np
import time
import random


def get_image_from_detector():
    """ Collect image from detector

    Actually this just produces a random image
    """
    # TODO: obtain used synchrotron from EBay
    return np.random.random((2000, 2000))

##########################################################
# Some image processing functions from our collaborators #
##########################################################

def process_1(img):
    time.sleep(random.random())
    return img


def process_2(img):
    time.sleep(random.random() / 2)
    return img / 10


def process_3(img_1, img_2):
    time.sleep(random.random() / 2)
    return img_1 + img_2


def save_to_database(img):
    time.sleep(0.5)

"""
Dear parallel programmer,

Please make the following happen on every image that we detect.

x = process_1(img)
y = process_2(img)
z = process_3(x, y)
save_to_database(img)
save_to_database(z)

Sincerely,
Beam Scientist
"""

def collect_from_beam():
    """ Collect data from beam, submit processing tasks """
    client = get_client()
    while True:
        delay = sleep_time.get()  # wait for photons to collect
        time.sleep(delay)

        local_image = get_image_from_detector()  # this is a numpy array
        remote_image = client.scatter(local_image, direct=True)

        result_1 = client.submit(process_1, remote_image)
        result_2 = client.submit(process_2, remote_image)

        merged_image = client.submit(process_3, result_1, result_2)

        save_raw_image = client.submit(save_to_database, remote_image)
        save_final_image = client.submit(save_to_database, merged_image)

        fire_and_forget([save_raw_image, save_final_image])


if __name__ == '__main__':
    client = Client('localhost:8786')
    sleep_time = Variable()
    sleep_time.set(2)

    # Long running tasks that feed images into the cluster
    futures = [client.submit(collect_from_beam, pure=False, workers='beam-1'),
               client.submit(collect_from_beam, pure=False, workers='beam-2')]
