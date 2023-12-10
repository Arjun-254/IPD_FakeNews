from googlesearch import search
from bs4 import BeautifulSoup
import requests
import pandas as pd

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


                currentp=""
                # Scrape <p> tags
                p_tags = soup.find_all('p')
                for p in p_tags:
                    currentp+=p.text
                article_content.append(currentp)


                currenth1=""
                # Scrape <h1> tags
                h1_tags = soup.find_all('h1')
                for h1 in h1_tags:
                   currenth1+=h1.text

                article_titles.append(currenth1)

                
            except Exception as e:
                print(f"An error occurred: {e}")
    return article_titles, article_content


# Get the titles and contents
def get_title_and_content(search_query_results):
    titles, contents = get_title_and_content(search_query_results)

    # Create a pandas DataFrame
    data = {'Title': titles, 'Content': contents}
    df = pd.DataFrame(data)
    return df




