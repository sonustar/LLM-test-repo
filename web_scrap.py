import requests
from bs4 import BeautifulSoup
import json

# Define the URL of the Hyperledger Fabric documentation page you want to scrape
url = 'https://hyperledger-fabric.readthedocs.io/en/latest/'  

def fetch_html(url):
    """Fetch HTML content from a URL."""
    try:
        # sending the GET request !! 
        response = requests.get(url)
        # helps to see the status whether we are getting the response or not !!
        response.raise_for_status() 
        # returning the text !!
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_paragraphs(html):
    """Extract paragraphs from HTML content."""
    soup = BeautifulSoup(html, 'html.parser')
    # find all the paragraphs with p tag : 
    paragraphs = soup.find_all('p')  
    return [p.get_text() for p in paragraphs if len(p.get_text().strip()) > 10]

def save_to_json(paragraphs, filename='rtdocs2.json'):
    """Save paragraphs to a JSON file."""
    data = [{'input': paragraph} for paragraph in paragraphs]
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    html_content = fetch_html(url)
    if html_content:
        paragraphs = extract_paragraphs(html_content)
        print(type(paragraphs))
        save_to_json(paragraphs)  
        print(f"Saved {len(paragraphs)} paragraphs to rtdocs.json")

if __name__ == "__main__":
    main()
