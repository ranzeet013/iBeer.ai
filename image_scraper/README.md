## Images Scraper: Beer Image Downloader

A Python script to scrape and download images from Google Images using a custom search query.

### Key Features
- **Custom Search Query:** Fetch images based on any keyword (default: "beer images").
- **Configurable Image Count:** Specify how many images you want to download.
- **Custom Save Location:** Save images in any directory with organized file names.
- **Duplicate Handling:** Ensures no duplicate image URLs are downloaded.
- **Automatic Pagination:** Scrapes images across multiple pages.

### Arguments Explained

- `-s` or `--search`: Search term for the images (default: "beer images").
- `-n` or `--num_images`: Number of images to download (default: 10).
- `-d` or `--directory`: Directory to save images (default: `./beer_images`).

### How It Works
1. The script sends requests to Google Images using your search query.
2. Extracts image URLs from the HTML response using BeautifulSoup.
3. Downloads images and saves them locally with unique file names.

### Notes
- **Delays:** Random delays between requests to prevent being flagged by Google.
- **Responsibility:** Use the script responsibly, adhering to Google's Terms of Service.
- **Image Quality:** Only downloads images that have valid `src` attributes.

### Example Usage

```bash
python google_image_scraper.py --search "craft beer" --num_images 20 --directory ./craft_beer_images
```

This example will download 20 images related to "craft beer" and save them in the `./craft_beer_images` directory.

### Prerequisites
- Python 3.7 or higher
- Required Python libraries:
  - `requests`
  - `BeautifulSoup4`
  - `os`
  - `argparse`

Install the required libraries using:
```bash
pip install requests beautifulsoup4
```
