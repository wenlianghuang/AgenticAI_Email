import wikipedia

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for a given query and return the summary of the first result.
    
    Parameters:
    - query (str): The search term to look up on Wikipedia.
    
    Returns:
    - str: The summary of the first Wikipedia article found for the query.
    """
    try:
        # Set the language for English Wikipedia
        wikipedia.set_lang("en")
        # Search Wikipedia and get the summary of the first result
        summary = wikipedia.summary(query, sentences=3)
        return f"✅ {summary} STOP [NEXT]"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query is ambiguous. Here are some suggestions: {e.options}"
    except wikipedia.exceptions.PageError as e:
        return "The page does not exist on Wikipedia."
    except Exception as e:
        return f"An error occurred: {e}"