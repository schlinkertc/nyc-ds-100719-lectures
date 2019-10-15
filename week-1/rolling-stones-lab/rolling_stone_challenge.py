# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv

with open('data.csv') as f:
    reader = csv.DictReader(f)
    roll_dict = [{k:v 
                  for k,v in row.items()
                  } for row in reader
                ]
    #try to do the same thing in series of nested loops instead of comprehension
    roll_dict1 =[]
    for row in reader:
        roll_dict1.append(row)
        print(type(row))
        #for k,v in row.items():
            #roll_dict1.append(dict())
    
def find_by_name(album_name):
    for album in roll_dict:
        if album['album'] == album_name:
            return album
    return None

def find_by_rank(album_rank):
    for album in roll_dict:
        if album['number'] == str(album_rank):
            return album
    return "None"

def find_by_year(year):
    albums = [album for album in roll_dict if album['year'] == str(year)]
    return albums 

def find_by_years(start_year,end_year):
    albums = [album for album in roll_dict if int(album['year']) 
    in range(int(start_year),int(end_year)+1)]
    return albums 

def find_by_ranks(start_rank,end_rank):
    albums = [album for album in roll_dict if int(album['number']) 
    in range(int(start_rank),int(end_rank)+1)]
    return albums 

def all_titles():
    all_titles = [album['album'] for album in roll_dict]
    return sorted(all_titles)

#never have function name equal to a list name. 
all_titles = all_titles()

def all_artists():
    all_artists = [album['artist'] for album in roll_dict]
    return sorted(all_artists)

all_artists = all_artists()
unique_artists = list(set(all_artists))


album_count = [{album['artist']:all_artists.count(album['artist'])} for album in roll_dict]
    
unique_album_count = []
for artist in unique_artists:
    unique_album_count.append(dict(Artist=artist,Albums=all_artists.count(artist)))
unique_album_count = sorted(unique_album_count,key=lambda i: i['Albums'],reverse=True)

#list_words_comp = try to write list_words as a list comprehension
list_words = []
for title in all_titles:
    for words in title.split():
        list_words.append(words.lower())  

word_counts = []
for word in list(set(list_words)):
    word_counts.append(dict(Word=word,Count=list_words.count(word)))
word_counts = sorted(word_counts,key=lambda i: i['Count'],reverse=True)

import matplotlib.pyplot as plt
x = [int(album['year']) for album in roll_dict]
x1 = []
for album in roll_dict:
    x1.append(int(album['year']))
#x = sorted(x)
decades = plt.hist(x,range=(1950,2020),bins=7)

genres = [album['genre'] for album in roll_dict]
parsed_genres = []
for genre in genres:
    parsed_genres.append(genre.split(',')[0])
#parsed_genres = list(set(parsed_genres))

import pandas as pd
pd.Series(parsed_genres).value_counts().plot('bar')

text_file = open('top-500-songs.txt','r')

