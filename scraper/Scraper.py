# import os
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urlencode
# from google.colab import drive

# # Mount Google Drive to save images
# drive.mount('/content/drive')

# # URL template for Google Images search
# GOOGLE_IMAGE_URL = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'


# def extract_images(query, quantity, save_dir):
#     """
#     Downloads a specified number of images from Google Images based on a search query 
#     and saves them to a specified directory in Google Drive.

#     Parameters:
#     query (str): The search term for the images (e.g., 'tuborg image').
#     quantity (int): The number of images to download.
#     save_dir (str): The directory path where the images will be saved in Google Drive.

#     Example:
#     extract_images('tuborg image', 10, '/content/drive/MyDrive/bottles/Tuborg')
#     """
#     # Construct the search URL using the query
#     search_url = GOOGLE_IMAGE_URL + urlencode({'q': query})
#     print(f'Fetching from URL: {search_url}')
    
#     # Fetch the page content from Google Images
#     response = requests.get(search_url)
#     response.raise_for_status()  # Check if the request was successful
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Find all image tags on the page
#     img_tags = soup.find_all('img')
    
#     # Create the directory to save images if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Initialize the image count
#     count = 0
#     print('Please wait...')
    
#     for img_tag in img_tags:
#         if count >= quantity:
#             break
        
#         # Get the image URL from the 'src' attribute
#         img_url = img_tag.get('src')
#         if img_url:
#             try:
#                 # Determine the file extension based on the image URL
#                 if img_url.endswith('.png'):
#                     ext = '.png'
#                 elif img_url.endswith('.jpg'):
#                     ext = '.jpg'
#                 elif img_url.endswith('.jfif'):
#                     ext = '.jfif'
#                 elif img_url.endswith('.svg'):
#                     ext = '.svg'
#                 else:
#                     ext = '.jpg'  # Default to .jpg if the extension is unknown
                
#                 # Download the image content
#                 img_response = requests.get(img_url, stream=True)
#                 img_response.raise_for_status()  # Check if the request was successful
                
#                 # Save the image to the specified directory
#                 filename = os.path.join(save_dir, f'{count}{ext}')
#                 with open(filename, 'wb') as file:
#                     for chunk in img_response.iter_content(chunk_size=8192):
#                         file.write(chunk)
                
#                 count += 1
#             except Exception as e:
#                 print(f'Failed to download image {img_url}: {e}')
    
#     print(f'Downloaded {count} images successfully in folder "{save_dir}"!')


# if __name__ == "__main__":
#     query = 'tuborg image'  # Default search term for the images
#     quantity = int(input('How many photos do you want? '))  # User input for the number of images
#     save_dir = '/content/drive/MyDrive/bottles/Tuborg'  # Directory path in Google Drive to save images
#     extract_images(query, quantity, save_dir)



import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import time
import random

# URL template for Google Images search
GOOGLE_IMAGE_URL = 'https://www.google.com/search?site=&tbm=isch&source=hp&'

def extract_images(query, quantity, save_dir):
    """
    Downloads a specified number of unique images from Google Images based on a search query 
    and saves them to a specified directory on the local machine.

    Parameters:
    query (str): The search term for the images (e.g., 'tuborg image').
    quantity (int): The number of images to download.
    save_dir (str): The directory path where the images will be saved on the local machine.
    """
    # Create the directory to save images if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    unique_image_urls = set()
    count = 0
    page = 0  # To track pagination

    print('Please wait...')
    
    while count < quantity:
        # Construct the search URL with pagination
        search_url = GOOGLE_IMAGE_URL + urlencode({'q': query, 'start': page * 20})
        print(f'Fetching from URL: {search_url}')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        # Fetch the page content from Google Images
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all image tags on the page
        img_tags = soup.find_all('img')
        
        for img_tag in img_tags:
            if count >= quantity:
                break
            
            img_url = img_tag.get('src')
            if img_url and img_url not in unique_image_urls:
                unique_image_urls.add(img_url)  # Track unique URLs
                
                try:
                    # Download the image content
                    img_response = requests.get(img_url, stream=True, headers=headers)
                    img_response.raise_for_status()  # Check if the request was successful
                    
                    # Determine the file extension
                    ext = os.path.splitext(img_url)[1] if os.path.splitext(img_url)[1] else '.jpg'
                    
                    # Save the image to the specified directory
                    filename = os.path.join(save_dir, f'{count}{ext}')
                    with open(filename, 'wb') as file:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    
                    count += 1
                    print(f'Downloaded image {count}: {img_url}')
                    
                    # Wait a random amount of time to avoid being blocked
                    time.sleep(random.uniform(2, 4))
                except Exception as e:
                    print(f'Failed to download image {img_url}: {e}')
        
        page += 1  # Move to the next page of results

        # If not enough unique images were found, wait before retrying
        if count < quantity:
            print('Not enough unique images found on this page. Fetching more images...')
            time.sleep(random.uniform(5, 10))  # Wait before the next request

    print(f'Downloaded {count} unique images successfully in folder "{save_dir}"!')

if __name__ == "__main__":
    query = 'Nepal Ice Beer image'  # Default search term for the images
    quantity = int(input('How many photos do you want? '))  # User input for the number of images
    save_dir = 'C:/Users/User/OneDrive/Desktop/iBeer.ai/Nepalice'  # Directory path on local machine to save images
    extract_images(query, quantity, save_dir)

