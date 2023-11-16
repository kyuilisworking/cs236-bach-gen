import requests
from bs4 import BeautifulSoup
import os

# URL of the website
url = 'https://www.bachcentral.com/midiindexcomplete.html'

# Send a request to the website
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all MIDI file links
midi_links = soup.find_all('a', href=True)

# Directory to save the files
save_directory = '../data/midi_files'
os.makedirs(save_directory, exist_ok=True)

for link in midi_links:
    href = link['href']
    # Check if the link is a MIDI file
    if href.endswith('.mid'):
        # Construct the full URL
        download_url = url.rsplit('/', 1)[0] + '/' + href

        # Get the MIDI file
        midi_response = requests.get(download_url)
        midi_response.raise_for_status()

        # Save the file
        filename = os.path.join(save_directory, href.split('/')[-1])
        with open(filename, 'wb') as file:
            file.write(midi_response.content)

        print(f'Downloaded {filename}')
