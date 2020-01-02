TABLES = {}
TABLES['businesses'] =  ("""CREATE TABLE businesses 
                        (id varchar (50) not null unique, 
                        name varchar (50) not null, 
                        review_count int(30) not null, 
                        rating float(20) not null,
                        price float (20) not null,
                        PRIMARY KEY (id))
                        engine=innoDB ;
                        """)

TABLES['reviews'] = ("""CREATE TABLE Reviews
                    (r_id varchar (50) not null,
                    review_text longblob not null,
                    time_created datetime not null,
                    PRIMARY KEY(r_id),
                    FOREIGN KEY (id) REFERNCES (businesses.id)
                    )engine=innoDB;""")

# Get business info from Yelp

def yelp_API_call_bk_chinese(SEARCH_LIMIT):    
    term = 'Chinese'
    location = 'Brooklyn NY'
    offset = 0

    url = 'https://api.yelp.com/v3/businesses/search'
    headers = {'Authorization': 'Bearer {}'.format(config.api_key)}
    url_params = {'term':term.replace(' ','+'),'location':location.replace(' ','+'),'limit':50, 'offset':offset}

    response = requests.get(url, headers=headers, params=url_params)
    
    results = json.loads(response.content)
    
    return results

#parse business info from Yelp 

def parse_business_results(api_call):
    df = pd.DataFrame(api_call['businesses'])
    df = df[businesses_columns]
    df['price'] = df['price'].map(lambda x: len(x),na_action='ignore')
    df['price'].fillna(0,inplace=True)
    #df['coordinates']=df['coordinates'].map(lambda x: tuple(x.values()))
    return [tuple(x) for x in df.values]

# Insert parsed info to MySQL database 

business_stmt = """INSERT INTO businesses (id, name, review_count, rating, price) VALUES (%s, %s, %s, %s, %s)"""

def business_agg(SEARCH_LIMIT):
    cursor.executemany(business_stmt,parse_business_results(yelp_API_call_bk_chinese(SEARCH_LIMIT)))
    cnx.commit()
    return cursor.execute("""SELECT * FROM businesses;""")

# retreive business IDs from MySQL Database 

def businessid_query():
    cursor.execute('''select id from businesses''')
    result = cursor.fetchall()
    result = [x[0] for x in result]
    return result

# businessid_query returns a list of business IDs from MySQL.
# Pass that list through the Yelp API review function 

def yelp_api_call_reviews(r_ids):
    outputs = []
    for r_id in r_ids:
        url = 'https://api.yelp.com/v3/businesses/{}/reviews'.format(r_id)
        headers = {'Authorization': 'Bearer {}'.format(config.api_key)}

        response = requests.get(url, headers=headers)
    
        outputs.append(response.content)
        
        outputs = [json.loads(x) for x in outputs]
        
        #parse the outputs 
        dfs = []
    
        for item in outputs:
            dfs.append(pd.DataFrame(item['reviews']))
            df = pd.concat(dfs)
            df = df[['id','text','time_created']]
            df['business_id'] = r_id
            
            # return a list of tuples to add to reviews table in MySQL
            return [tuple(x) for x in df.values]

reviews_stmt = """INSERT INTO reviews (id, text, time_created, business_id) VALUES (%s, %s, %s, %s)"""

## write a function to insert reviews info into reviews table. Aggregate functions as neccesary...

