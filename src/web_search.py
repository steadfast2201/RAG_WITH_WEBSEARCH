import asyncio
import aiohttp
import os
import urllib.parse
from googlesearch import search
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from config_settings import REQUESTS_HEADER  # Custom headers for HTTP requests


# Taking in queries (usually from an LLM like LLaMA or GPT)
# Running those queries on search engines (Google or DuckDuckGo)
# Downloading the web pages asynchronously
# Extracting the visible body text
# Saving the text to local files (in a ./downloaded folder)
# Cleaning up files afterward (optional)


# Encode URL into a safe filename format
# Converts any URL into a filesystem-safe filename by percent-encoding it.
# Example:
#     Input: https://example.com/page?q=test
#     Output: https%3A%2F%2Fexample.com%2Fpage%3Fq%3Dtest
def encode_url_to_filename(url):
    return urllib.parse.quote(url, safe="")


# Decode filename back into the original URL
# This is the core async fetcher:
#     Uses aiohttp to make a GET request to the url
#     Parses the HTML with BeautifulSoup
#     Extracts only the visible text from the <body> tag
#     Writes it into a .txt file named using the encoded URL
def decode_filename_to_url(filename):
    return urllib.parse.unquote(filename)


# -----------------------
# Web Fetching Logic
# -----------------------

# Asynchronously fetch the HTML content of a URL, extract visible text from <body>, and save it to a file
async def fetch_and_save(session, url, folder):
    print(url)
    try:
        filename = encode_url_to_filename(url)
        filepath = os.path.join(folder, filename)

        async with session.get(url) as response:
            response.raise_for_status()  # Raise exception for non-200 responses
            content = await response.text()
            soup = BeautifulSoup(content, "html.parser")
            text = soup.find("body").get_text(strip=True)  # Extract visible body text # type:ignore
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"Saved: {filepath}")
    except Exception:
        return f"Failed to fetch or save for url: {url}"


# Get search result URLs from the selected provider
def get_urls(query: str, num_results: int, provider: str):
    if provider == "google":
        return search(query, num_results=num_results, lang="en", region="us")
    elif provider == "duckduckgo":
        ddgs = DDGS()
        return [
            url.get("href")
            for url in ddgs.text(query, max_results=num_results, region="us-en")
        ]


# -----------------------
# Main Fetch Controller
# -----------------------


# Given a list of queries, fetch top search results and save their page content locally
async def fetch_web_pages(
    queries: list[str],
    num_results: int,
    provider: str,
    download_dir: str = "./downloaded",
):
    os.makedirs(download_dir, exist_ok=True)  # Create download directory if not exists
    for query in queries:
        urls = get_urls(query, num_results, provider)

        async with aiohttp.ClientSession(headers=REQUESTS_HEADER) as session:
            tasks = [fetch_and_save(session, url, download_dir) for url in urls] # type:ignore
            await asyncio.gather(*tasks)  # Run all fetch tasks concurrently


# -----------------------
# Cleanup Function
# -----------------------


# Remove all files from the download directory
def remove_temp_files(download_dir: str = "./downloaded"):
    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        os.remove(file_path)
