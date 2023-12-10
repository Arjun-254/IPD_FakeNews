from googlesearch import search
from bs4 import BeautifulSoup
import requests

def get_top_links(query, num_links=10):
    linksgot =[]
    try:
        # Perform Google search and get the top links
        search_results = search(query, num_results=num_links)

        # Print the top links
        print(f"Top {num_links} links for '{query}':")
        for i, link in enumerate(search_results, start=1):
            print(f"{i}. {link}")
            linksgot.append(link)
    except Exception as e:
        print(f"An error occurred: {e}")
    return linksgot

# Example usage:
query_to_search = "Sam Altman Was fired From OpenAI"
search_query_results = get_top_links(query_to_search)

# Scrape <p> and <h1> tags from the first search result
def get_title_and_content(search_query_results):
    article_titles =[]
    article_content =[]
    if search_query_results:
        for results in search_query_results:
            try:
                # Send a request to the URL and get the HTML content
                response = requests.get(results)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Scrape <p> tags
                p_tags = soup.find_all('p')
                for p in p_tags:
                    article_content.append(p.text)

                # Scrape <h1> tags
                h1_tags = soup.find_all('h1')
                for h1 in h1_tags:
                    article_titles.append(h1.text)
            except Exception as e:
                print(f"An error occurred: {e}")
    return article_titles, article_content




