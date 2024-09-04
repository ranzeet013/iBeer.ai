# Image Scraper
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Image Scraper is a Python script designed to automate the process of downloading images from Google Images based on a specific search query. This tool is particularly useful for users who need to quickly gather a collection of images related to a particular topic, whether for research, personal projects, or any other purpose where a set of visual resources is required.

## Key Features

- **Search Query**: Allows users to specify the exact term or topic they are interested in. For example, you can search for "tuborg beer" to find images related to that specific type of beer.
- **Customizable Number of Photos**: Users can define how many images they want to download, providing flexibility depending on their needs.
- **Save Directory**: You can choose the directory on your local machine where the downloaded images will be stored, ensuring that you have organized and easily accessible files.

## How It Works

1. **Search Query Input**: The script prompts the user to enter a search term. This query is used to perform a search on Google Images.

2. **Number of Images**: The user specifies how many images they want to download. The script will attempt to download the specified number of images related to the search query.

3. **Save Directory**: The user provides the path to a directory where the images will be saved. The script ensures that all downloaded images are stored in this location.

4. **Image Downloading**: The script sends requests to Google Images, retrieves the image URLs, and downloads the images. It handles the image retrieval process, including managing different image formats and ensuring that the images are correctly saved to the specified directory.

## Prerequisites

Before running the script, ensure you have the following installed:

- **Python**: Version 3.6 or higher.
- **pip**: The Python package installer.

## Setup Instructions

1. **Clone or Download the Repository**  
   Start by cloning this repository or downloading the script to your local machine.

   ```bash
   git clone https://github.com/ranzeet013/iBeer.ai.git
   cd scraper

2. **Create a Virtual Environment**

   First, create a virtual environment to isolate your project's dependencies:

   ```bash
   python -m venv venv

3. **Activate the Virtual Environment (Windows)**

   Activate the virtual environment with the following command:

   ```bash
   venv\Scripts\activate

4. **Run the Script**

   To start the image scraper, run the image_scraper.py script. The script will prompt     you to enter the search query, the number of images to download, and the directory where the images will be saved.

   ```bash
   python Scraper.py


