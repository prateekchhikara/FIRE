import json, os, pickle
from tqdm import tqdm


with open("dataset.json", "r") as f:
    data = json.load(f)


train_IDs, test_IDs, valid_IDs = [], [], []
layer1Data = []
for k in tqdm(data):
    if k["partition"] == "train":
        train_IDs.append(k["id"])
    if k["partition"] == "test":
        test_IDs.append(k["id"])
    if k["partition"] == "val":
        valid_IDs.append(k["id"])
        
    layer1Data.append(k)
    
print(len(valid_IDs), len(train_IDs), len(test_IDs), len(layer1Data))
    
import urllib.request, requests
    
import multiprocessing
def download_image(element): 
    """ Download image from url and save it to disk.

    Args:
        element (dictionary): instance of the dataset
    """
    output_dir = "train"
    if element["id"] in train_IDs:
        foodId = element["id"]
        allImages = element["images"]
        for im in allImages:
            try:
                response = requests.get(im["url"], stream=True)
                with open(f"{output_dir}/{foodId}/{im['id']}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except:
                pass
    
    output_dir = "test"     
    if element["id"] in test_IDs:
        foodId = element["id"]
        allImages = element["images"]
        for im in allImages:
            try:
                response = requests.get(im["url"], stream=True)
                with open(f"{output_dir}/{foodId}/{im['id']}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except:
                pass
    
    output_dir = "val"         
    if element["id"] in valid_IDs:
        foodId = element["id"]
        allImages = element["images"]
        for im in allImages:
            try:
                response = requests.get(im["url"], stream=True)
                with open(f"{output_dir}/{foodId}/{im['id']}", "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            except:
                pass


print("downloading started")
def download_images(dataLayer2, num_processes=40):
    """Download images using multiprocessing.

    Args:
        dataLayer2 (json): dataset
        num_processes (int, optional): number of processes. Defaults to 40.
    """
    with multiprocessing.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(download_image, [(element) for element in dataLayer2])):
            pass
        
download_images(data, num_processes=40)
print("download completed")