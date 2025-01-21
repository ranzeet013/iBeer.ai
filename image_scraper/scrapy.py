import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import time
import random
import argparse

# URL template for Google Images search
GOOGLE_IMAGE_URL = 'https://www.google.com/search?site=&tbm=isch&source=hp&'

def fetch_image_urls(query, quantity, headers, start=0):
    """
    Fetches image URLs from Google Images based on the query.

    Parameters:
    query (str): Search term for the images (e.g., 'beer images').
    quantity (int): Number of images to download.
    headers (dict): HTTP headers for the request.
    start (int): Pagination start index.

    Returns:
    set: A set of unique image URLs.
    """
    unique_image_urls = set()
    while len(unique_image_urls) < quantity:
        search_url = GOOGLE_IMAGE_URL + urlencode({'q': query, 'start': start})
        print(f'Fetching from URL: {search_url}')
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')

        for img_tag in img_tags:
            img_url = img_tag.get('src')
            if img_url and img_url.startswith('http') and img_url not in unique_image_urls:
                unique_image_urls.add(img_url)
                if len(unique_image_urls) >= quantity:
                    break

        start += 20
        time.sleep(random.uniform(2, 5))  

    return unique_image_urls

def download_images(image_urls, save_dir):
    """
    Downloads images from the provided URLs.

    Parameters:
    image_urls (set): Set of image URLs.
    save_dir (str): Directory to save the images.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
            ext = os.path.splitext(img_url)[1] or '.jpg'
            file_path = os.path.join(save_dir, f'beer_{i+1}{ext}')
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(8192):
                    file.write(chunk)
            print(f'Downloaded: {file_path}')
            time.sleep(random.uniform(1, 3))  
        except Exception as e:
            print(f'Failed to download {img_url}: {e}')

def main():
    parser = argparse.ArgumentParser(description="Scrape beer images from Google Images.")
    parser.add_argument('-s', '--search', default='beer images', type=str, help='Search term for images')
    parser.add_argument('-n', '--num_images', default=10, type=int, help='Number of images to download')
    parser.add_argument('-d', '--directory', default='./beer_images', type=str, help='Directory to save images')
    args = parser.parse_args()

    query = args.search
    quantity = args.num_images
    save_dir = args.directory

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    print("Starting image scraping...")
    image_urls = fetch_image_urls(query, quantity, headers)
    print(f"Found {len(image_urls)} images. Downloading...")
    download_images(image_urls, save_dir)
    print(f"Images downloaded to {save_dir}.")

if __name__ == '__main__':
    main()
