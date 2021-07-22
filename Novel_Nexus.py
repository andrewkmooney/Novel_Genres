import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from utils import process_text
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import FreqDist, text
from wordcloud import WordCloud
import plotly.express as px
from PIL import Image


filename_1 = 'data/genre_model.sav'
filename_2 = 'data/age_model.sav'
filename_3 = 'data/genre_vectorizer.sav'
filename_4 = 'data/age_vectorizer.sav'

genre_vectorizer = pickle.load(open(filename_3, 'rb'))
genre_model = pickle.load(open(filename_1, 'rb'))
age_model = pickle.load(open(filename_2, 'rb'))
age_vectorizer = pickle.load(open(filename_4, 'rb'))

df_genre = pd.read_csv('data/vector_df.csv')

df_genre['TF_Vector'] = np.load('data/Book_Vectors.npy',allow_pickle=True)

df_vectors = df_genre[[
    'Book_Title',
    'Author_Name', 
    'Year_published', 
    'Book_Description', 
    'TF_Vector',
    'Rating_votes',
    'Review_number',
    'Rating_score',
    'Age_Group']]

genre_list = ['Fantasy',
 'Romance',
 'Adventure',
 'Dystopia',
 'Paranormal',
 'Space Opera',
 'Aliens',
 'Historical',
 'Space',
 'Time Travel',
 'Speculative',
 'Apocalyptic',
 'War',
 'Mystery',
 'Steampunk',
 'Horror',
 'Queer']

def find_most_similar(description, number=10, year=1900, detail=None):
    
    '''
    Takes a book description, processes it into a vector and returns the most similar books based on the cosine
    similarity of that vector to the other book descriptions. Can return single descriptions or a table of books.
    
    Parameters
    ----------
    
    description - string - a book description that we want to analyze in the form of a string. May include \n, which will be replaced with blank spaces.
    number - int - determines the number of rows returned in the final dataframe
    year - int - restricts the comparisons to books published after a certain date
    detail - None or int - if None is passed, the return will be a dataframe. If an integer is passed, the dataframe will return the details of that row of the returned dataframe
    
    Returns
    -------
    
    A dataframe ordered by the cosine similarity of the book descriptions. If an integer is passed in 'detail'
    then the return will be the dataframe as well as the book description of that row of the dataframe.
    
    '''
    
    df = df_vectors[df_vectors['Year_published'] >= year]
    
    vector = create_tf_vector(description, model='genre')
    df['Similarity'] = df['TF_Vector'].map(lambda x: cosine_similarity(vector, x)[0][0])
    
    df_final = df.sort_values(by=['Similarity'], ascending=False)
    
    df_final.drop(columns=['TF_Vector'],inplace=True)
    
    if detail == None:
        return df_final
    else:
        title = df_final['Book_Title'].iloc[detail]
        author = df_final['Author_Name'].iloc[detail]
        year = df_final['Year_published'].iloc[detail]
        desc = df_final['Book_Description'].iloc[detail]
        
        st.write(f'''
        Most similar #{detail+1}:
        
        Title: {title}   Author: {author}  Year: {year}
        
        {desc}
        ''')
        return df_final

def print_scores(description, mode='genre'):
    
    '''
    Prints the results of either the genre or the age model when a book description is passed through.
    The function preprocesses the description with a vectorizer based on which setting is desired and then
    feeds that vector into either the genre or age model, returning the results as a dataframe with the
    results as probabilities and sorted by value.
    
    Parameters
    ----------
    
    description - string - a book description that we want to analyze in the form of a string. May include \n, which will be replaced with blank spaces.
    model - string - "genre", "age" - determines which vectorizer to use, either the age vectorizer or the genre vectorizer
    
    Returns
    -------
    
    A dataframe of the scores, listed in descending order based on probability
    
    '''
    
    if mode == 'genre':
        labels = genre_list # Uses the list of genres that were used to train the genre model
        model = genre_model
    elif mode == 'age':
        labels = ['Adult', 'Children', 'Young Adult']
        model = age_model
    
    # Vectorizes the description and then creates prediction probabilities
    vector = create_tf_vector(description, model=mode)
    predict = model.predict_proba(vector) 
    
    # Creates score dictionary that will be turned into a dataframe
    score_dict = {}
    
    if mode =='genre':
        for (n, label) in enumerate(labels):
            score_dict[label] = predict[n][0][1]
            
    elif mode =='age':
        for (n, label) in enumerate(labels):
            score_dict[label] = predict[0][n]

    # Sorts the dictionary based on score values
    final_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Turns dictionary into a dataframe
    df = pd.DataFrame.from_dict([final_dict]).T
    
    df.columns = ['Probabilities']
    
    return df

def create_tf_vector(description, model='genre'):
    
    '''
    Uses the process_text function to pre-process a book description in order to feed it into the appropriate model.
    
    Parameters
    ----------
    
    description - string - a book description that we want to analyze in the form of a string. May include \n, which will be replaced with blank spaces.
    model - string - "genre", "age" - determines which vectorizer to use, either the age vectorizer or the genre vectorizer
    
    Returns
    -------
    
    A vectorized prepresentation of the text as an numpy array
    
    '''
    
    if model == 'genre':
        vectorizer = genre_vectorizer
    elif model == 'age':
        vectorizer = age_vectorizer
        
    description.replace('\n','') # Removes any superfluous line breaks
    tokens = process_text(description, keep_stopwords=False) # Removes stop words and lemmatizes text
    vector = vectorizer.transform([' '.join(tokens)]) # Turns the tokens into a vector with the appropriate vectorizer
    
    return vector.toarray()

def most_impactful_words(description, num=None):
    
    '''
    Finds the most impactful words in a description. It vectorizes the description and then sorts the words in terms
    of vector magnitude.
    
    Parameters
    ----------
    
    description - string - the book description that you would like to analyze as a single string
    
    num - int - number of features you would like to see. Defaults to all relevant words. 
                If an integer greater than the number of words is passed, the entire word list is returned.
                
    Returns
    -------
    
    A dataframe showing the words in order of importance
    
    '''
    
    vector = create_tf_vector(description)
    words = genre_vectorizer.get_feature_names()
    
    df = pd.DataFrame(vector, columns = words).T
    
    df.columns = ['Value']
    
    final_df = df[df['Value'] > 0].sort_values(by='Value', ascending=False)
    
    if (num == None) or (num >= len(final_df)):
        return final_df
    else:
        return final_df.head(num)

image = Image.open('images/sci_fi_scape.jpeg')

st.image(image)

st.title('Novel Nexus')

add_selectbox = st.sidebar.selectbox(
    "Page",
    ("Genre Analysis", "About", "The Data")
)

if add_selectbox == 'Genre Analysis':

    st.header('Helping Writers Refine their Query Letters')

    st.text('''
    Welcome to Novel Nexus, an app designed to help new Science Fiction Authors get published. 
    For more information on how this all works and why you should use it, 
    check out my About Page.
    ''')

    st.subheader('Genre and Age Group Identifier')
    st.text('''
    Input a Science Fiction Novel Description here:
    ''')

    text_dict = {}

    book_title = st.text_input('Title of Novel')
    txt = st.text_area('Text to analyze', height=300)

    processed = False

    start = st.selectbox('Select Starting Year',options=np.array([2020, 2015, 2010, 2005, 2000, 1995, 1990, 1900]))

    if st.button('Process'):
        processed = True
    
    if processed == True:
        index = len(text_dict)
        vector = create_tf_vector(txt)
        age_df = print_scores(txt, mode='age')
        genre_df = print_scores(txt, mode='genre')
        words = most_impactful_words(txt)

        text_dict[index] = {
            'title': book_title,
            'vector': vector,
            'age': age_df,
            'genre': genre_df,
            'words': words,
        }

    if processed == True:

        st.title(f'''
            Data for {book_title}
            ''')
        col1, col2 = st.beta_columns(2)
        with col1:
            st.subheader('Sub-Genre Scores')
            st.write(text_dict[index]['genre'])
        with col2:
            st.subheader('Important Words')
            st.write(text_dict[index]['words'])
        st.subheader('Age Group Scores')
        st.write(text_dict[index]['age'])
        fig = px.bar(text_dict[index]['genre'], title=f'Sub-Genre Scores for {book_title}', labels={'value':'Score','index':'Sub-Genre'})
        st.plotly_chart(fig)
        fig = px.bar(text_dict[index]['age'], title=f'Age Group Scores for {book_title}',labels={'index':'Age Group','value':'Score'})
        st.plotly_chart(fig)

        

        books = find_most_similar(txt,year=start, detail=1)

        st.write(books)

        st.selectbox('Pick a Book to Explore', options=books['Book_Title'])


    # Figure out how to get detail on a single book without rerunning the whole thing
    # Caching?
    # Add in Comparison functions
    # Figure out Text Highlighting

    # 

        # elif choice == 'Most Impactful Words':
        #     st.write(text_dict[index]['words'])

    # genre = st.radio(
    #     "What's your favorite movie genre",
    #     ('Comedy', 'Drama', 'Documentary'))

    # if genre == 'Comedy':
    #     st.write('You selected comedy.')
    # else:
    #     st.write("You didn't select comedy.")

    # options = st.multiselect(
    #     'What are your favorite colors',
    #     ['Green', 'Yellow', 'Red', 'Blue'],
    #     ['Yellow', 'Red'])

    # st.write('You selected:', options)

    # import datetime as dt
    # appointment = st.slider(
    #     "Schedule your appointment:",
    #     value=(dt.date(1900,1,1), dt.date(2021,3,15)))

    # st.write("You're scheduled for:", appointment)

    # start_color, end_color = st.select_slider(
    #     'Select a range of color wavelength',
    #     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
    #     value=('red', 'blue'))
    # st.write('You selected wavelengths between', start_color, 'and', end_color)

    # txt = st.text_area('Text to analyze', '''
    # ...     It was the best of times, it was the worst of times, it was
    # ...     the age of wisdom, it was the age of foolishness, it was
    # ...     the epoch of belief, it was the epoch of incredulity, it
    # ...     was the season of Light, it was the season of Darkness, it
    # ...     was the spring of hope, it was the winter of despair, (...)
    # ...     ''')
    # st.write('Sentiment:', 'Good')

elif add_selectbox == 'About':

    st.header('About This Project')

    st.text('''
        Welcome to Novel Nexus! This is a completely open source project designed by me, 
        Andrew Mooney, an aspiring sci-fi/fantasy novelist and professional data scientist.
        I created this tool as a part of my final Capstone project for the Flatiron School's 
        ata Science Bootcamp, but I intend to continue updating it over time. This project 
        serves two major functions: 1) to help new writers build the best descriptions of 
        their stories in order to help them refine cover letters to send to agents and 
        2) to help them identify other novels that might have similar themes and or styles.

        For the casual reader, it serves as a pandora-style novel recommender. If you have 
        read any of the books in my training dataset, I can recommend the top 10 similar books.
    ''')

    st.subheader('Who is This For?')

    st.text('''
        I have been writing since I was about 10 years old, and I was obsessed with becoming a
        published author. Since I graduated college in 2008 I've been attempting to break into
        the industry with the help of several kind, excellent authors willing to share their
        knowledge. For aspiring authors, it can be almost impossible to figure out HOW to start
        your journey towards becoming a professional writer. It's easy to think it relies 
        entirely on the quality and content of the book, but in order to find an agent excited
        about your work you need to write an impressive cover letter.

        There are many excellent resources online regarding what a novel cover letter should
        include and I'll link to them in the resources section of this page. I was lucky enough 
        to take a course with Jac Jemc, author of "The Grip of It" (an excellent horror novel
        if you're into literary ghost stories) and there are a few key things I discovered. Not
        only should you write a succinct and compelling blurb (similar to that on a book jacket)
        but you should also mention a list of books similar to your own. 
        
        This list of recommendations can also help in identifying the TYPES of agents you want
        to submit to. Yes, we can all say we wrote something similar to Game of Thrones, but I'm
        pretty sure George R.R. Martin's agent isn't looking for any new talent. That said, books
        SIMILAR to A Game of Thrones might have agents that ARE looking for new talent.

        Also, if you read a good sci-fi book and want a recommendation on what might be a good
        follow up, my recommender can do that easily.
        ''')

    st.subheader('How does it work?')

    st.text('''
        For a full in-depth explanation along with the source code, feel free to visit my github
        repo with a full Jupyter Notebook of my exploration and findings. But for people who don't
        read Data Science blogs for fun, here are the basics:

        I took around 12000 science fiction books scraped from Goodreads which I found thanks to
        Kaggle (thank you!). Each book came with author info, ratings, publication history and,
        importantly, a book jacket description. Each book has been tagged by Goodreads users which
        I turned into the following major sci-fi genres:

        - Fantasy
        - Romance
        - Adventure
        - Dystopia
        - Paranormal
        - Space Opera
        - Aliens
        - Historical
        - Space
        - Time Travel
        - Speculative
        - Apocalyptic
        - War
        - Mystery
        - Steampunk
        - Horror
        - Queer

        Every novel has multiple genre tags and here is the frequency of each tag in the original
        dataset:
        ''')

    st.text('''
        Using Natural Language Processing, I turned each of the descriptions into what is called 
        a 'vector' (I'll get into that in a second) and then I fed these vectors into a machine 
        learning algorithm that predicted the genre labels on each one. When you put your book
        description into this website, it turns it into a vector and then predicts which genres
        your book belongs to, which age group it is aimed at and what are the top 10 books it's 
        most similar to.
        ''')

    st.header('What Does "Vectorization" Mean?')

    st.text('''
        The big question with any model using Natural Language Processing (NLP) is: how the heck 
        does it work? People have a bad habit of saying 'Machine Learning' waving their hand and 
        expecting SkyNet to take over and eliminate the human race. The reality is much more 
        mundane and technical. In essence, Machine Learning is the process of turning real world 
        things into a string of numbers (called a vector or a 'tensor' to be more accurate). Once 
        we've got the real world thing in number form, we can use a series of tools to multiply 
        those numbers by 'variable weights' to try to get a correct answer. And that's...about it. 
        Each 'trained' model is essentially just a series of weights that, when multiplied by your 
        description vector, tells us how close it is to certain desired labels.

        Why am I telling you this? Because I want you to understand that algorithms, while amazing, 
        are stupid. They don't think like humans, they don't act like humans, they are just a 
        series of numbers that help turn specific groups of numbers into a smaller set of specific 
        numbers. As the old quote from George E. P. Box goes: "Essentially, all models are wrong, 
        but some are useful."

        So, one of the most important things to understand about how this model works isn't how the 
        magic math on the backend works, but how does it turn your book description into a vector? 
        There are two main methods that I used: SpaCy and TF-IDF.
    ''')

    st.subheader('SpaCy and Skip-Grams')

    st.text('''
        English is hard. Anyone who has ever had to learn it as a second language knows how confusing 
        and backwards this mess of a language truly is. Take this classic example of the (debatably) 
        optional Oxford Comma:

        IMAGE OF OXFORD COMMA

        As I said before, machines are dumb, but there are some very clever ways to work with it. 
        When NLP was first becoming a major research subject in the 70s and 80s, computer scientists 
        worked closely with linguists to see if there were ways to capture (in numbers) the nuances 
        of sentence structure. Is it more helpful to know if a word is a noun or a verb? What if it's 
        the subject or the object of a sentence? What if the word is a proper noun versus a regular 
        noun? 

        Enter SpaCy and Google's Skip-Gram algorithm. Using a machine learning model trained on 100 
        million Wikipedia articles, Google developed an algorithm that sought to improve the accuracy 
        of word vectorization. Skip-grams are the basis of major text generation models because it 
        identifies the underlying structure of words in addition to their frequency. Essentially, 
        the goal of the project was to make sure that if: king is to queen as man is to woman. And, 
        amazingly, it works pretty well!

        When SpaCy vectorizes your text, it identifies the word in terms of part of speech, if it is 
        an 'entity', like a person or a country, and turns the text into one vector that has 96 
        values. Why 96? I don't know, but that's how it does it. Unfortunately, the SpaCy vectors 
        really didn't perform well in the predictive model, but they can be useful for identifing 
        similar descriptions using a very effective method called 'cosine similarity'.
    ''')

    st.subheader('Term Frequency - Inverse Document Frequency')

    st.text('''
        Honestly, I really wanted to like SpaCy simply because it sounds SO COOL on paper. The 
        reality is that it didn't do very well in terms of predicting genre labels or age groups. 
        The method that worked exceptionally well (in comparison) was an old school method called 
        'Term Frequency-Inverse Document Frequency' or 'TF-IDF' for short. Here's how it works: it 
        takes ALL of the descriptions and figures out what are the most common words in the full 
        corpus. I then removed uninteresting words like 'the', 'and', 'but', etc. so we can see 
        what words are most useful. Here's a word cloud of the word frequency:
        ''')

    st.text('''
        And here's a bar graph showing the most frequent words:
    ''')

    st.text('''
        As you can see, some words like 'war' will show up in miltary story descriptions more 
        frequently than in, say, a romance young adult book. TF-IDF takes the most common 400 
        words and figures out how often those 400 words show up in some descriptions and, 
        importantly, DON'T show up in other ones. But those 400 words are ALL that it cares
        about. It can also identify pairs of words. Outer Space might be more frequent than the
        words space and outer alone. But beyond that, it doesn't care about sentence structure,
        metaphors, or anything else that makes writing so magical. Just word frequency. Bit
        of a bummer right? But hey, that's math for you.''')

    st.subheader('Why Should You Care?')

    st.text('''
        Because it's janky! I know it, and you should know it too. I firmly believe that we 
        shouldn't see Machine Learning as magic.It is flawed, problematic, and inaccurate...
        but in the end it can produce some useful insights. This site will show you which 
        words increase the description's genre and age scores so you can see how different 
        drafts give different scores. 

        For the similarity table, these are supposed to point you in the right direction
        so you can find other books that MIGHT be similar. The similarity scores are also
        moderately stupid. There is no machine learning involved, it just figures out the
        similarity between two of the vectors and then ranks the descriptions based on those
        scores. This is not magic, it's not anything you should stake your career on. 
        That said, if you're struggling to break into the indusrty and need a boost, hopefully
        this will get you started!

        For an in-depth analysis of the underlying data, please check out my 'The Data' page.
    ''')

    st.header('Resources')

elif add_selectbox == 'The Data':

    st.header('The Data')

    st.text('''
        As I mentioned in the 'About' section, which I highly recommend you read before this
        page, the data feeding this algorithm was pulled from Goodreads and compiled on the
        machine learning repository website Kaggle.com. Here is a snapshot of the raw data:
    ''')

    # RAW DATAFRAME

    st.text('''
        There were about 15000 books in the original data, but I removed the bottom 25 percent
        of books that had the fewest reviews. Some of the books are short story collections,
        but most are bona fide science fiction novels. Here are the most popular novels:
    ''')


